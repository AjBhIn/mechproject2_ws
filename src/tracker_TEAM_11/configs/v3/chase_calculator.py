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

COSTMAP_MARGIN_CELLS = 6        
MAX_SAFE_COST = 220              

CAPTURE_ORIENT_DIST = 0.35      
MIN_YAW_UPDATE_DIST = 0.15      
FLANK_ENGAGE_DIST = 1.8         

# --- TACTICAL LEASH & OVERSHOOT VARIABLES ---
# Approach leash during CHASING
CHASE_LEASH_DIST = 0.8          

# VECTOR OVERSHOOT: During STRIKING, project the goal 1.0m BEHIND the enemy.
# This tricks Nav2 into driving at full speed THROUGH the enemy's inflation layer,
# allowing the robot to cross the 0.50m capture threshold before Nav2 can brake!
STRIKE_OVERSHOOT_DIST = 1.0     

FLANK_CONE_DEG = 60.0           
FLANK_OFFSET_DIST = 1.5         
MAX_YAW_RATE_SMOOTHING = 1.0    

class ChaserCalculator(Node):
    def __init__(self):
        super().__init__('chaser_calculator')

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

        self.current_state = 'CHASING'
        self.costmap = None
        self.last_valid_yaw = 0.0

        self.last_enemy_yaw = None
        self.last_enemy_time = None
        self.enemy_yaw_rate = 0.0
        self.filtered_target_x = None
        self.filtered_target_y = None

        self.flank_active_logged = False
        self.dive_active_logged = False
        self.was_sleeping = False  

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        state_topic = f'/{self.robot_ns}/robot_state' if self.robot_ns else '/robot_state'
        costmap_topic = f'/{self.robot_ns}/global_costmap/costmap' if self.robot_ns else '/global_costmap/costmap'

        self.create_subscription(String, state_topic, self.state_cb, 10)
        self.create_subscription(OccupancyGrid, costmap_topic, self.costmap_cb, 10)

        self.timer = self.create_timer(LOOP_RATE_SEC, self.control_loop)
        self.get_logger().info("[CHASER CALCULATOR] Initialized. Vector Overshoot Active for Inflation Bypass!")

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

    def project_to_free_space(self, target_x: float, target_y: float, p_x: float, p_y: float):
        if self.is_pose_safe(target_x, target_y, max_cost=MAX_SAFE_COST):
            return target_x, target_y

        best_tx, best_ty = target_x, target_y
        min_dist_to_pursuer = float('inf')
        found_safe = False

        for r in [0.15, 0.3, 0.45, 0.6]:
            for angle in range(0, 360, 45):
                rad = math.radians(angle)
                tx = target_x + r * math.cos(rad)
                ty = target_y + r * math.sin(rad)
                
                if self.is_pose_safe(tx, ty, max_cost=MAX_SAFE_COST):
                    d_to_p = math.hypot(tx - p_x, ty - p_y)
                    if d_to_p < min_dist_to_pursuer:
                        min_dist_to_pursuer = d_to_p
                        best_tx, best_ty = tx, ty
                        found_safe = True
            
            if found_safe:
                return best_tx, best_ty
                
        return target_x, target_y

    def update_enemy_yaw_rate(self, enemy_yaw: float, current_time: float):
        if self.last_enemy_yaw is not None and self.last_enemy_time is not None:
            dt = current_time - self.last_enemy_time
            if 0.01 < dt < 1.0:
                raw_rate = abs(math.atan2(math.sin(enemy_yaw - self.last_enemy_yaw), math.cos(enemy_yaw - self.last_enemy_yaw))) / dt
                self.enemy_yaw_rate = 0.6 * self.enemy_yaw_rate + 0.4 * raw_rate
        self.last_enemy_yaw = enemy_yaw
        self.last_enemy_time = current_time

    def control_loop(self):
        offensive_states = ['CHASING', 'STRIKING', 'COOLDOWN']
        
        if self.current_state not in offensive_states:
            if not self.was_sleeping:
                self.get_logger().info(f"[CHASER CALCULATOR] Yielding to Evasion Node.")
                self.was_sleeping = True
                
            self.filtered_target_x = None
            self.filtered_target_y = None
            self.flank_active_logged = False
            self.dive_active_logged = False
            return

        if self.was_sleeping:
            self.get_logger().info("[CHASER CALCULATOR] Resuming offensive pursuit.")
            self.was_sleeping = False

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

        tactical_x, tactical_y = 0.0, 0.0

        # RULE 1: Anti-Ramming Dodge
        if front_cone_diff < math.radians(FLANK_CONE_DEG) and dist < FLANK_ENGAGE_DIST:
            if not self.flank_active_logged:
                self.get_logger().warn(f"[CHASER CALCULATOR] Anti-Ramming! Flanking {FLANK_OFFSET_DIST}m around enemy front.")
                self.flank_active_logged = True
            self.dive_active_logged = False
                
            flank_side = (angle_to_pursuer_from_enemy - enemy_yaw) > 0
            flank_offset_angle = enemy_yaw + (math.pi / 2.0 if flank_side else -math.pi / 2.0)
            tactical_x = e_x + FLANK_OFFSET_DIST * math.cos(flank_offset_angle)
            tactical_y = e_y + FLANK_OFFSET_DIST * math.sin(flank_offset_angle)
            
        else:
            self.flank_active_logged = False
            
            # RULE 2: Strike Dive (VECTOR OVERSHOOT)
            if self.current_state == 'STRIKING':
                if not self.dive_active_logged:
                    self.get_logger().info(f"[CHASER CALCULATOR] STRIKING! Projecting goal {STRIKE_OVERSHOOT_DIST}m OVERSHOOT behind enemy to force full-speed penetration through inflation layer!")
                    self.dive_active_logged = True
                
                # Project the goal 1.0m BEHIND the enemy along the approach line.
                # Nav2 sees a clear goal beyond the inflation zone and maintains velocity.
                approach_angle = math.atan2(e_y - p_x, e_x - p_x)
                tactical_x = e_x + (STRIKE_OVERSHOOT_DIST * math.cos(enemy_yaw))
                tactical_y = e_y + (STRIKE_OVERSHOOT_DIST * math.sin(enemy_yaw))
                
            # RULE 3: Approach Phase
            else:
                self.dive_active_logged = False
                tactical_x = self.filtered_target_x
                tactical_y = self.filtered_target_y

        target_x, target_y = 0.0, 0.0

        if self.current_state == 'STRIKING':
            # Drop leash during strike: Goal is set far behind enemy to maintain drive speed
            target_x = tactical_x
            target_y = tactical_y
        else:
            # Apply 0.8m leash during approach
            dist_to_tactical = math.hypot(tactical_x - p_x, tactical_y - p_y)
            angle_to_tactical = math.atan2(tactical_y - p_y, tactical_x - p_x)
            
            if dist_to_tactical > CHASE_LEASH_DIST:
                target_x = p_x + (CHASE_LEASH_DIST * math.cos(angle_to_tactical))
                target_y = p_y + (CHASE_LEASH_DIST * math.sin(angle_to_tactical))
            else:
                target_x = tactical_x
                target_y = tactical_y

        if dist > MIN_YAW_UPDATE_DIST:
            self.last_valid_yaw = math.atan2(dy, dx)

        if dist <= CAPTURE_ORIENT_DIST:
            rot_z = c_rot.z
            rot_w = c_rot.w
        else:
            rot_z = math.sin(self.last_valid_yaw / 2.0)
            rot_w = math.cos(self.last_valid_yaw / 2.0)

        safe_x, safe_y = self.project_to_free_space(target_x, target_y, p_x, p_y)

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