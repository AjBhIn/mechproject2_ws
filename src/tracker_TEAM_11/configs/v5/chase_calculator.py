#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
import tf2_ros
from geometry_msgs.msg import TransformStamped
import numpy as np

# --- GLOBAL TUNING VARIABLES ---
LOOP_RATE_SEC = 0.033

COSTMAP_MARGIN_CELLS = 6        
MAX_SAFE_COST = 85              

CAPTURE_ORIENT_DIST = 0.35      
MIN_YAW_UPDATE_DIST = 0.15      

# Tactical Leash & Precision Strike
CHASE_LEASH_DIST = 0.8          
STRIKE_TARGET_DIST = 0.40       
ORBITAL_FLANK_DIST = 1.2        # Distance to maintain while sweeping around the enemy's front

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
        self.map_data = None
        self.map_res = 0.05
        self.map_width = self.map_height = 0
        self.map_origin_x = self.map_origin_y = 0.0

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
        self.get_logger().info("[CHASER CALCULATOR] Dynamic Orbital Flanking and Strict Path Validation Active.")

    def state_cb(self, msg):
        self.current_state = msg.data

    def costmap_cb(self, msg: OccupancyGrid):
        self.costmap = msg
        self.map_data = np.array(msg.data, dtype=np.int16)
        self.map_res = msg.info.resolution
        self.map_width, self.map_height = msg.info.width, msg.info.height
        self.map_origin_x, self.map_origin_y = msg.info.origin.position.x, msg.info.origin.position.y

    @staticmethod
    def get_yaw_from_quaternion(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def get_cell_cost(self, x: float, y: float) -> int:
        if self.map_data is None: 
            return 100
        raw_col = int((x - self.map_origin_x) / self.map_res)
        raw_row = int((y - self.map_origin_y) / self.map_res)

        if raw_col < COSTMAP_MARGIN_CELLS or raw_col >= (self.map_width - COSTMAP_MARGIN_CELLS) or \
           raw_row < COSTMAP_MARGIN_CELLS or raw_row >= (self.map_height - COSTMAP_MARGIN_CELLS):
            return 100

        col = max(0, min(raw_col, self.map_width - 1))
        row = max(0, min(raw_row, self.map_height - 1))
        cost = self.map_data[row * self.map_width + col]
        return 100 if cost == -1 else cost

    def truncate_to_line_of_sight(self, p_x: float, p_y: float, target_x: float, target_y: float):
        dist = math.hypot(target_x - p_x, target_y - p_y)
        
        if dist < 0.05 or self.current_state == 'STRIKING':
            return target_x, target_y

        step_size = 0.05
        steps = max(2, int(dist / step_size))
        last_safe_x, last_safe_y = p_x, p_y
        
        for t in np.linspace(0.0, 1.0, steps):
            tx = p_x + t * (target_x - p_x)
            ty = p_y + t * (target_y - p_y)
            if self.get_cell_cost(tx, ty) > MAX_SAFE_COST:
                return last_safe_x, last_safe_y
            last_safe_x, last_safe_y = tx, ty
            
        return target_x, target_y

    def find_arc_path_to_target(self, target_x: float, target_y: float, p_x: float, p_y: float, our_yaw: float):
        if self.current_state == 'STRIKING':
            return target_x, target_y

        # Fix: Check the ENTIRE line to the target, not just the destination coordinate.
        trunc_x, trunc_y = self.truncate_to_line_of_sight(p_x, p_y, target_x, target_y)
        if math.hypot(trunc_x - target_x, trunc_y - target_y) < 0.05:
            return target_x, target_y

        # If straight line is blocked, arc around the obstacle
        best_tx, best_ty = trunc_x, trunc_y
        best_score = float('inf')

        for r in [0.2, 0.4, 0.6, 0.8]:
            for angle in range(0, 360, 30):
                rad = math.radians(angle)
                tx = target_x + r * math.cos(rad)
                ty = target_y + r * math.sin(rad)
                
                angle_from_robot = math.atan2(ty - p_y, tx - p_x)
                turn_diff = abs(math.atan2(math.sin(angle_from_robot - our_yaw), math.cos(angle_from_robot - our_yaw)))
                
                # Forward-only lock to prevent reversing
                if turn_diff > math.radians(100.0):
                    continue
                
                # Check the line TO the arc point to ensure it's not cutting through a wall
                arc_trunc_x, arc_trunc_y = self.truncate_to_line_of_sight(p_x, p_y, tx, ty)
                if math.hypot(arc_trunc_x - tx, arc_trunc_y - ty) < 0.05:
                    d_to_target = math.hypot(tx - target_x, ty - target_y)
                    d_to_robot = math.hypot(tx - p_x, ty - p_y)
                    score = d_to_target + (d_to_robot * 0.2)
                    
                    if score < best_score:
                        best_score = score
                        best_tx, best_ty = tx, ty
                        
        return best_tx, best_ty

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
        now_sec = self.get_clock().now().nanoseconds / 1e9
        
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
        our_yaw = self.get_yaw_from_quaternion(t_our.transform.rotation)
        
        c_x, c_y = t_carrot.transform.translation.x, t_carrot.transform.translation.y
        c_rot = t_carrot.transform.rotation

        e_x, e_y = t_enemy.transform.translation.x, t_enemy.transform.translation.y
        enemy_yaw = self.get_yaw_from_quaternion(t_enemy.transform.rotation)

        self.update_enemy_yaw_rate(enemy_yaw, now_sec)

        if self.filtered_target_x is None:
            self.filtered_target_x, self.filtered_target_y = c_x, c_y
        else:
            alpha = 0.4 if self.enemy_yaw_rate > MAX_YAW_RATE_SMOOTHING else 0.8  
            self.filtered_target_x = alpha * c_x + (1.0 - alpha) * self.filtered_target_x
            self.filtered_target_y = alpha * c_y + (1.0 - alpha) * self.filtered_target_y

        angle_to_pursuer_from_enemy = math.atan2(p_y - e_y, p_x - e_x)
        rel_angle_from_front = math.atan2(math.sin(angle_to_pursuer_from_enemy - enemy_yaw), math.cos(angle_to_pursuer_from_enemy - enemy_yaw))
        abs_front_diff = abs(rel_angle_from_front)

        tactical_x, tactical_y = 0.0, 0.0

        # --- DYNAMIC ORBITAL FLANKING ---
        # If the robot is in the front or side (not in the rear 120-degree pocket), sweep around it safely.
        if abs_front_diff < math.radians(120.0):
            if not self.flank_active_logged:
                self.get_logger().warn(f"[CHASER CALCULATOR] Target in front/side. Orbiting {ORBITAL_FLANK_DIST}m around to the rear pocket!")
                self.flank_active_logged = True
            self.dive_active_logged = False
            
            # Determine which way is shorter to the back (left or right)
            direction_sign = 1.0 if rel_angle_from_front >= 0.0 else -1.0
            
            # Project the target 60 degrees further towards the rear along the orbital circle
            orbit_angle = angle_to_pursuer_from_enemy + (direction_sign * math.radians(60.0))
            
            tactical_x = e_x + ORBITAL_FLANK_DIST * math.cos(orbit_angle)
            tactical_y = e_y + ORBITAL_FLANK_DIST * math.sin(orbit_angle)
            
        else:
            self.flank_active_logged = False
            
            # --- REAR POCKET REACHED: STRIKE / APPROACH ---
            if self.current_state == 'STRIKING':
                if not self.dive_active_logged:
                    self.get_logger().info(f"[CHASER CALCULATOR] STRIKING! Precision targeting {STRIKE_TARGET_DIST}m in front of enemy to avoid crashing.")
                    self.dive_active_logged = True
                
                approach_angle = math.atan2(e_y - p_y, e_x - p_x)
                tactical_x = e_x - (STRIKE_TARGET_DIST * math.cos(approach_angle))
                tactical_y = e_y - (STRIKE_TARGET_DIST * math.sin(approach_angle))
                
            else:
                self.dive_active_logged = False
                tactical_x = self.filtered_target_x
                tactical_y = self.filtered_target_y

        target_x, target_y = 0.0, 0.0

        # Apply Leash Logic
        if self.current_state == 'STRIKING':
            target_x = tactical_x
            target_y = tactical_y
        else:
            dist_to_tactical = math.hypot(tactical_x - p_x, tactical_y - p_y)
            angle_to_tactical = math.atan2(tactical_y - p_y, tactical_x - p_x)
            
            if dist_to_tactical > CHASE_LEASH_DIST:
                target_x = p_x + (CHASE_LEASH_DIST * math.cos(angle_to_tactical))
                target_y = p_y + (CHASE_LEASH_DIST * math.sin(angle_to_tactical))
            else:
                target_x = tactical_x
                target_y = tactical_y

        # Update Orientation
        dx = target_x - p_x
        dy = target_y - p_y
        dist_to_target = math.hypot(dx, dy)

        if dist_to_target > MIN_YAW_UPDATE_DIST:
            self.last_valid_yaw = math.atan2(dy, dx)

        if math.hypot(e_x - p_x, e_y - p_y) <= CAPTURE_ORIENT_DIST:
            rot_z = c_rot.z
            rot_w = c_rot.w
        else:
            rot_z = math.sin(self.last_valid_yaw / 2.0)
            rot_w = math.cos(self.last_valid_yaw / 2.0)

        safe_x, safe_y = self.find_arc_path_to_target(target_x, target_y, p_x, p_y, our_yaw)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.broadcast_frame
        t.transform.translation.x = safe_x
        t.transform.translation.y = safe_y
        t.transform.translation.z = 0.0
        t.transform.rotation.z = rot_z
        t.transform.rotation.w = rot_w

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ChaserCalculator())
    rclpy.shutdown()

if __name__ == '__main__':
    main()