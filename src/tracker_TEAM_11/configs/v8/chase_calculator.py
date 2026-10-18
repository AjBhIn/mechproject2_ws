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
MAX_EMERGENCY_COST = 95         
STUCK_ESCAPE_DIST = 1.2         

CAPTURE_ORIENT_DIST = 0.35      
MIN_YAW_UPDATE_DIST = 0.15      

# Tactical Leash & Precision Strike
CHASE_LEASH_DIST = 0.8          
STRIKE_TARGET_DIST = 0.40       
ORBITAL_FLANK_DIST = 1.2        

MAX_YAW_RATE_SMOOTHING = 1.0    
LOG_THROTTLE_SEC = 1.0          

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
        self.last_log_time = 0.0
        
        # State lock to prevent micro-stuttering/wiggling at the 85-cost boundary
        self.is_recovering = False

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        state_topic = f'/{self.robot_ns}/robot_state' if self.robot_ns else '/robot_state'
        costmap_topic = f'/{self.robot_ns}/global_costmap/costmap' if self.robot_ns else '/global_costmap/costmap'

        self.create_subscription(String, state_topic, self.state_cb, 10)
        self.create_subscription(OccupancyGrid, costmap_topic, self.costmap_cb, 10)

        self.timer = self.create_timer(LOOP_RATE_SEC, self.control_loop)
        self.get_logger().info("[CHASER CALCULATOR] Initialized with Anti-Wiggle Lock and Forward Recovery.")

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

    def get_cell_costs_vectorized(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        if self.map_data is None: 
            return np.full(xs.shape, 100)
        
        raw_cols = ((xs - self.map_origin_x) / self.map_res).astype(np.int32)
        raw_rows = ((ys - self.map_origin_y) / self.map_res).astype(np.int32)
        
        invalid = (raw_cols < COSTMAP_MARGIN_CELLS) | (raw_cols >= (self.map_width - COSTMAP_MARGIN_CELLS)) | \
                  (raw_rows < COSTMAP_MARGIN_CELLS) | (raw_rows >= (self.map_height - COSTMAP_MARGIN_CELLS))
                  
        cols = np.clip(raw_cols, 0, self.map_width - 1)
        rows = np.clip(raw_rows, 0, self.map_height - 1)
        
        costs = self.map_data[rows * self.map_width + cols]
        costs = np.where((costs == -1) | invalid, 100, costs)
        return costs

    def is_pose_safe(self, x: float, y: float, max_cost: int = MAX_SAFE_COST) -> bool:
        if self.costmap is None: return True
        return self.get_cell_cost(x, y) <= max_cost

    def get_chaser_stuck_recovery_target(self, our_x: float, our_y: float, our_yaw: float, 
                                         enemy_x: float, enemy_y: float, carrot_x: float, carrot_y: float, 
                                         current_cost: int, should_log: bool):
        probe_angles = np.linspace(0, 2 * math.pi, 36, endpoint=False)
        probe_x = our_x + 0.6 * np.cos(probe_angles)
        probe_y = our_y + 0.6 * np.sin(probe_angles)
        probe_costs = self.get_cell_costs_vectorized(probe_x, probe_y)

        # 1. Gradient Descent (Escape the wall)
        cost_drop = current_cost - probe_costs
        scores = cost_drop * 15.0

        # 2. Avoid Enemy Center (Restored 75-degree cone to prevent ramming physical corners)
        angle_to_enemy = math.atan2(enemy_y - our_y, enemy_x - our_x)
        enemy_diff = np.abs(np.arctan2(np.sin(probe_angles - angle_to_enemy), np.cos(probe_angles - angle_to_enemy)))
        scores = np.where(enemy_diff < np.deg2rad(75.0), scores - 500.0, scores)

        # 3. Stay Offensive (Move toward the rear pocket target)
        dist_to_carrot = np.hypot(probe_x - carrot_x, probe_y - carrot_y)
        scores -= dist_to_carrot * 5.0

        valid_angles = probe_costs < MAX_EMERGENCY_COST

        if should_log:
            debug_costs = [f"{int(math.degrees(ang)) % 360}°:{cost}" for ang, cost in zip(probe_angles, probe_costs)]
            self.get_logger().info(f"[CHASER STUCK SWEEP] Robot Cost: {current_cost} | Surrounding: [{', '.join(debug_costs)}]")

        if not np.any(valid_angles):
            # Boxed in! Calculate escape vector and orient forward to drive out
            cross_product = math.cos(our_yaw) * math.sin(angle_to_enemy) - math.sin(our_yaw) * math.cos(angle_to_enemy)
            spin_dir = -1.0 if cross_product > 0 else 1.0
            best_escape_yaw = our_yaw + (spin_dir * math.pi / 2.0)
            
            spin_x = our_x + 0.6 * math.cos(best_escape_yaw)
            spin_y = our_y + 0.6 * math.sin(best_escape_yaw)
            return spin_x, spin_y, best_escape_yaw

        best_idx = np.argmax(np.where(valid_angles, scores, -1e9))
        best_escape_yaw = float(probe_angles[best_idx])

        # Project goal outward into free space (Restored 0.45m force field to prevent projecting inside enemy)
        escape_dist = 0.6 
        for d in (1.2, 1.0, 0.8, 0.6):
            tx = our_x + d * math.cos(best_escape_yaw)
            ty = our_y + d * math.sin(best_escape_yaw)
            if self.get_cell_cost(tx, ty) < MAX_SAFE_COST and math.hypot(tx - enemy_x, ty - enemy_y) >= 0.45:
                escape_dist = d
                break

        rec_x = our_x + escape_dist * math.cos(best_escape_yaw)
        rec_y = our_y + escape_dist * math.sin(best_escape_yaw)

        if should_log:
            self.get_logger().warn(
                f"[CHASER STUCK RECOVERY] Driving forward out via {math.degrees(best_escape_yaw):.1f}°. "
                f"Targeting {escape_dist:.2f}m."
            )

        # Return best_escape_yaw so the robot turns and drives forward into the clearing
        return rec_x, rec_y, best_escape_yaw

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

        trunc_x, trunc_y = self.truncate_to_line_of_sight(p_x, p_y, target_x, target_y)
        if math.hypot(trunc_x - target_x, trunc_y - target_y) < 0.05:
            return target_x, target_y

        best_tx, best_ty = trunc_x, trunc_y
        best_score = float('inf')

        for r in [0.2, 0.4, 0.6, 0.8]:
            for angle in range(0, 360, 30):
                rad = math.radians(angle)
                tx = target_x + r * math.cos(rad)
                ty = target_y + r * math.sin(rad)
                
                angle_from_robot = math.atan2(ty - p_y, tx - p_x)
                turn_diff = abs(math.atan2(math.sin(angle_from_robot - our_yaw), math.cos(angle_from_robot - our_yaw)))
                
                if turn_diff > math.radians(100.0):
                    continue
                
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
            self.is_recovering = False
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

        # --- TACTICAL TARGETING ---
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

        if abs_front_diff < math.radians(120.0):
            if not self.flank_active_logged:
                self.get_logger().warn(f"[CHASER CALCULATOR] Target in front/side. Orbiting {ORBITAL_FLANK_DIST}m around to the rear pocket!")
                self.flank_active_logged = True
            self.dive_active_logged = False
            
            direction_sign = 1.0 if rel_angle_from_front >= 0.0 else -1.0
            orbit_angle = angle_to_pursuer_from_enemy + (direction_sign * math.radians(60.0))
            
            tactical_x = e_x + ORBITAL_FLANK_DIST * math.cos(orbit_angle)
            tactical_y = e_y + ORBITAL_FLANK_DIST * math.sin(orbit_angle)
            
        else:
            self.flank_active_logged = False
            
            if self.current_state == 'STRIKING':
                if not self.dive_active_logged:
                    self.get_logger().info(f"[CHASER CALCULATOR] STRIKING! Precision targeting {STRIKE_TARGET_DIST}m in front of enemy.")
                    self.dive_active_logged = True
                
                approach_angle = math.atan2(e_y - p_y, e_x - p_x)
                tactical_x = e_x - (STRIKE_TARGET_DIST * math.cos(approach_angle))
                tactical_y = e_y - (STRIKE_TARGET_DIST * math.sin(approach_angle))
                
            else:
                self.dive_active_logged = False
                tactical_x = self.filtered_target_x
                tactical_y = self.filtered_target_y

        target_x, target_y = 0.0, 0.0

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

        dx = target_x - p_x
        dy = target_y - p_y
        dist_to_enemy = math.hypot(e_x - p_x, e_y - p_y)

        if math.hypot(dx, dy) > MIN_YAW_UPDATE_DIST:
            self.last_valid_yaw = math.atan2(dy, dx)

        if dist_to_enemy <= CAPTURE_ORIENT_DIST:
            rot_z = c_rot.z
            rot_w = c_rot.w
        else:
            rot_z = math.sin(self.last_valid_yaw / 2.0)
            rot_w = math.cos(self.last_valid_yaw / 2.0)

        safe_x, safe_y = self.find_arc_path_to_target(target_x, target_y, p_x, p_y, our_yaw)

        # --- DYNAMIC STUCK RECOVERY CHECK ---
        current_cost = self.get_cell_cost(p_x, p_y)
        front_x = p_x + 0.25 * math.cos(our_yaw)
        front_y = p_y + 0.25 * math.sin(our_yaw)
        front_cost = self.get_cell_cost(front_x, front_y)

        dist_to_safe = math.hypot(safe_x - p_x, safe_y - p_y)
        target_collapsed = (dist_to_safe < 0.25) and (dist_to_enemy > 0.60)
        
        # Lightweight spatial lock to prevent goal thrashing (wiggling) at the boundary
        if not self.is_recovering:
            if current_cost >= MAX_SAFE_COST or front_cost >= MAX_SAFE_COST or target_collapsed:
                self.is_recovering = True
        else:
            # Release recovery as soon as the robot backs up into 74 cost or below (maintains aggression)
            if current_cost < 75 and front_cost < 75 and not target_collapsed:
                self.is_recovering = False

        is_stuck = self.is_recovering and (self.current_state == 'CHASING')
        should_log = (now_sec - self.last_log_time > LOG_THROTTLE_SEC)

        if is_stuck:
            if should_log:
                reason = "High Costmap Elevation" if (current_cost >= MAX_SAFE_COST or front_cost >= MAX_SAFE_COST) else "Path Collapsed (Blocked)"
                self.get_logger().warn(f"[CHASER TRIGGER] Stuck detected via {reason}. Firing Recovery!")
                
            safe_x, safe_y, safe_yaw = self.get_chaser_stuck_recovery_target(
                p_x, p_y, our_yaw, e_x, e_y, c_x, c_y, current_cost, should_log
            )
            rot_z = math.sin(safe_yaw / 2.0)
            rot_w = math.cos(safe_yaw / 2.0)
            
            if should_log:
                self.last_log_time = now_sec

            self.filtered_target_x = None
            self.filtered_target_y = None

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