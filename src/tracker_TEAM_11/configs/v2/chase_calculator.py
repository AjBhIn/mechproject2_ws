#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
import tf2_ros
from geometry_msgs.msg import TransformStamped

# --- GLOBAL TUNING VARIABLES ---
LOOP_RATE_SEC = 0.033

# Costmap & Safety Boundaries
COSTMAP_MARGIN_CELLS = 6        
MAX_SAFE_COST = 80              
RAYCAST_STEPS = 20              

# Tracking & Targeting Distances
MAX_PROJECTION_DIST = 0.8       
CAPTURE_ORIENT_DIST = 0.35      
MIN_YAW_UPDATE_DIST = 0.15      
FLANK_ENGAGE_DIST = 1.8         

# Angles & Filters
FLANK_CONE_DEG = 60.0           
MAX_YAW_RATE_SMOOTHING = 1.0    

class ChaserCalculator(Node):
    """
    Dynamic interception path generator. Smooths target teleportation on sharp enemy turns 
    and uses an Anti-Ramming Flank Route to dodge head-on collisions.
    Continues to generate pursuit paths during CHASING, STRIKING, and COOLDOWN.
    """
    def __init__(self):
        super().__init__('chaser_calculator')

        # --- Dynamic Namespace Setup ---
        raw_ns = self.get_namespace().strip('/')
        default_robot_ns = raw_ns if raw_ns else 'our_bot'

        self.declare_parameter('chaser', default_robot_ns)
        self.declare_parameter('enemy', 'enemy_bot')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('pursuer_frame', '')
        self.declare_parameter('enemy_frame', '')
        self.declare_parameter('carrot_frame', '')
        self.declare_parameter('broadcast_frame', 'chase_nav_target')

        self.robot_ns = self.get_parameter('chaser').value.strip('/')
        self.enemy_ns = self.get_parameter('enemy').value.strip('/')
        self.map_frame = self.get_parameter('map_frame').value
        
        pursuer_p = self.get_parameter('pursuer_frame').value
        self.pursuer_frame = pursuer_p if pursuer_p else (f'{self.robot_ns}/base_link' if self.robot_ns else 'base_link')

        enemy_p = self.get_parameter('enemy_frame').value
        self.enemy_frame = enemy_p if enemy_p else (f'{self.enemy_ns}/base_link' if self.enemy_ns else 'base_link')

        carrot_p = self.get_parameter('carrot_frame').value
        self.carrot_frame = carrot_p if carrot_p else (f'{self.enemy_ns}/target_point' if self.enemy_ns else 'target_point')
        self.broadcast_frame = self.get_parameter('broadcast_frame').value

        # --- State Variables ---
        self.current_state = 'CHASING'
        self.costmap = None
        self.last_valid_yaw = 0.0

        # Spin Tracking & Target Smoothing
        self.last_enemy_yaw = None
        self.last_enemy_time = None
        self.enemy_yaw_rate = 0.0
        self.filtered_target_x = None
        self.filtered_target_y = None

        self.flank_active_logged = False

        # --- TF2 & Subscriptions ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        state_topic = f'/{self.robot_ns}/robot_state' if self.robot_ns else '/robot_state'
        costmap_topic = f'/{self.robot_ns}/global_costmap/costmap' if self.robot_ns else '/global_costmap/costmap'

        self.create_subscription(String, state_topic, self.state_cb, 10)
        self.create_subscription(OccupancyGrid, costmap_topic, self.costmap_cb, 10)

        # 30Hz Control loop
        self.timer = self.create_timer(LOOP_RATE_SEC, self.control_loop)
        self.get_logger().info("Smart Chaser Calculator initialized (Multi-Phase Pursuit Active).")

    def state_cb(self, msg):
        self.current_state = msg.data

    def costmap_cb(self, msg: OccupancyGrid):
        self.costmap = msg

    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def is_pose_safe(self, x: float, y: float, max_cost: int = MAX_SAFE_COST) -> bool:
        if self.costmap is None: return True
        info = self.costmap.info
        gx = int((x - info.origin.position.x) / info.resolution)
        gy = int((y - info.origin.position.y) / info.resolution)

        if gx < COSTMAP_MARGIN_CELLS or gx >= (info.width - COSTMAP_MARGIN_CELLS) or \
           gy < COSTMAP_MARGIN_CELLS or gy >= (info.height - COSTMAP_MARGIN_CELLS):
            return False

        cost = self.costmap.data[gy * info.width + gx]
        return 0 <= cost < max_cost

    def project_to_free_space(self, start_x: float, start_y: float, target_x: float, target_y: float):
        if self.is_pose_safe(target_x, target_y, max_cost=MAX_SAFE_COST):
            return target_x, target_y

        dist = math.hypot(target_x - start_x, target_y - start_y)
        if dist < 0.05: return start_x, start_y

        for i in range(1, RAYCAST_STEPS):
            alpha = 1.0 - (i / float(RAYCAST_STEPS))
            px = start_x + alpha * (target_x - start_x)
            py = start_y + alpha * (target_y - start_y)
            if self.is_pose_safe(px, py, max_cost=MAX_SAFE_COST):
                return px, py
                
        return start_x, start_y

    def update_enemy_yaw_rate(self, enemy_yaw: float, current_time: float):
        if self.last_enemy_yaw is not None and self.last_enemy_time is not None:
            dt = current_time - self.last_enemy_time
            if 0.01 < dt < 1.0:
                raw_rate = abs(math.atan2(math.sin(enemy_yaw - self.last_enemy_yaw), math.cos(enemy_yaw - self.last_enemy_yaw))) / dt
                self.enemy_yaw_rate = 0.6 * self.enemy_yaw_rate + 0.4 * raw_rate
        self.last_enemy_yaw = enemy_yaw
        self.last_enemy_time = current_time

    def control_loop(self):
        # We now keep chasing during all three offensive phases!
        offensive_states = ['CHASING', 'STRIKING', 'COOLDOWN']
        
        if self.current_state not in offensive_states:
            self.filtered_target_x = None
            self.filtered_target_y = None
            self.flank_active_logged = False
            return

        try:
            t_our = self.tf_buffer.lookup_transform(self.map_frame, self.pursuer_frame, rclpy.time.Time())
            t_carrot = self.tf_buffer.lookup_transform(self.map_frame, self.carrot_frame, rclpy.time.Time())
            t_enemy = self.tf_buffer.lookup_transform(self.map_frame, self.enemy_frame, rclpy.time.Time())
        except tf2_ros.TransformException:
            return

        p_x, p_y = t_our.transform.translation.x, t_our.transform.translation.y
        c_x, c_y = t_carrot.transform.translation.x, t_carrot.transform.translation.y
        c_rot = t_carrot.transform.rotation

        e_x, e_y = t_enemy.transform.translation.x, t_enemy.transform.translation.y
        enemy_yaw = self.get_yaw_from_quaternion(t_enemy.transform.rotation)

        now_sec = self.get_clock().now().nanoseconds / 1e9
        self.update_enemy_yaw_rate(enemy_yaw, now_sec)

        if self.filtered_target_x is None:
            self.filtered_target_x, self.filtered_target_y = c_x, c_y
        else:
            alpha = 0.4 if self.enemy_yaw_rate > MAX_YAW_RATE_SMOOTHING else 0.8  
            self.filtered_target_x = alpha * c_x + (1.0 - alpha) * self.filtered_target_x
            self.filtered_target_y = alpha * c_y + (1.0 - alpha) * self.filtered_target_y

        dx = self.filtered_target_x - p_x
        dy = self.filtered_target_y - p_y
        dist = math.hypot(dx, dy)

        angle_to_pursuer_from_enemy = math.atan2(p_y - e_y, p_x - e_x)
        front_cone_diff = abs(math.atan2(math.sin(angle_to_pursuer_from_enemy - enemy_yaw), math.cos(angle_to_pursuer_from_enemy - enemy_yaw)))

        if front_cone_diff < math.radians(FLANK_CONE_DEG) and dist < FLANK_ENGAGE_DIST:
            if not self.flank_active_logged:
                self.get_logger().info("Anti-Ramming Flank Route ACTIVE.")
                self.flank_active_logged = True
                
            flank_side = (angle_to_pursuer_from_enemy - enemy_yaw) > 0
            flank_offset_angle = enemy_yaw + (math.pi / 2.0 if flank_side else -math.pi / 2.0)
            target_x = e_x + 1.0 * math.cos(flank_offset_angle)
            target_y = e_y + 1.0 * math.sin(flank_offset_angle)
        else:
            self.flank_active_logged = False
            lead_dist = min(MAX_PROJECTION_DIST, max(0.4, dist * 0.7))
            if dist > lead_dist:
                ratio = lead_dist / dist
                target_x = p_x + dx * ratio
                target_y = p_y + dy * ratio
            else:
                target_x = self.filtered_target_x
                target_y = self.filtered_target_y

        if dist > MIN_YAW_UPDATE_DIST:
            self.last_valid_yaw = math.atan2(dy, dx)

        if dist <= CAPTURE_ORIENT_DIST:
            rot_z = c_rot.z
            rot_w = c_rot.w
        else:
            rot_z = math.sin(self.last_valid_yaw / 2.0)
            rot_w = math.cos(self.last_valid_yaw / 2.0)

        safe_x, safe_y = self.project_to_free_space(p_x, p_y, target_x, target_y)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.broadcast_frame
        t.transform.translation.x = safe_x
        t.transform.translation.y = safe_y
        t.transform.rotation.z = rot_z
        t.transform.rotation.w = rot_w

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ChaserCalculator())
    rclpy.shutdown()

if __name__ == '__main__':
    main()