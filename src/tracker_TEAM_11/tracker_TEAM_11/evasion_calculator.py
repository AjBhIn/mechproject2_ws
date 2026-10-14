#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String
import numpy as np

class EvasionCalculator(Node):
    """
    High-Performance Evasion node. Uses vectorized NumPy arrays to evaluate 144 escape paths 
    instantly, sorts them by safety/tactics, and only runs expensive raycasting on the Top 10.
    """
    def __init__(self):
        super().__init__('evasion_calculator')

        # --- Dynamic Namespace ---
        raw_ns = self.get_namespace().strip('/')
        default_robot_ns = raw_ns if raw_ns else 'our_bot'

        self.declare_parameter('chaser', default_robot_ns)
        self.declare_parameter('enemy', 'enemy_bot')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', '')
        self.declare_parameter('enemy_frame', '')
        self.declare_parameter('target_frame', 'dynamic_nav_target')

        self.robot_ns = self.get_parameter('chaser').value.strip('/')
        self.enemy_ns = self.get_parameter('enemy').value.strip('/')

        self.map_frame = self.get_parameter('map_frame').value
        robot_p = self.get_parameter('robot_frame').value
        self.robot_frame = robot_p if robot_p else (f'{self.robot_ns}/base_link' if self.robot_ns else 'base_link')

        enemy_p = self.get_parameter('enemy_frame').value
        self.enemy_frame = enemy_p if enemy_p else (f'{self.enemy_ns}/base_link' if self.enemy_ns else 'base_link')
        self.target_frame = self.get_parameter('target_frame').value
        
        # --- TF2 & Subscriptions ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.current_state = 'CHASING'
        
        state_topic = f'/{self.robot_ns}/robot_state' if self.robot_ns else '/robot_state'
        costmap_topic = f'/{self.robot_ns}/global_costmap/costmap' if self.robot_ns else '/global_costmap/costmap'

        self.state_sub = self.create_subscription(String, state_topic, self.state_cb, 10)
        self.costmap_sub = self.create_subscription(OccupancyGrid, costmap_topic, self.costmap_callback, 10)

        # --- Vectorized Candidate Grid (Precomputed) ---
        # 4 radii * 36 angles = 144 candidate points evaluated simultaneously
        self.sample_radii = np.array([0.6, 1.0, 1.5, 2.0])
        angles = np.deg2rad(np.arange(0, 360, 10))
        r_grid, a_grid = np.meshgrid(self.sample_radii, angles)
        
        self.cand_dx = (r_grid * np.cos(a_grid)).flatten()
        self.cand_dy = (r_grid * np.sin(a_grid)).flatten()
        self.cand_r = r_grid.flatten()

        # --- Map Variables & Hysteresis ---
        self.map_data = None
        self.map_res = 0.05
        self.map_width = self.map_height = 0
        self.map_origin_x = self.map_origin_y = 0.0 
        
        self.MAX_SAFE_COST = 40
        self.MIN_UPDATE_DIST = 0.02
        
        self.last_target_x, self.last_target_y = None, None
        self.active_target_x, self.active_target_y = None, None
        
        self.slide_direction_bias = None 
        self.last_chosen_angle = None 

        self.last_log_time = 0.0
        self.last_tf_time = 0.0 

        # 30Hz Overclock
        self.timer = self.create_timer(0.033, self.control_loop)
        self.get_logger().info("Ultra-Fast Robust EvasionCalculator initialized.")

    def state_cb(self, msg):
        self.current_state = msg.data

    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def costmap_callback(self, msg: OccupancyGrid):
        self.map_data = np.array(msg.data, dtype=np.int16)
        self.map_res = msg.info.resolution
        self.map_width, self.map_height = msg.info.width, msg.info.height
        self.map_origin_x, self.map_origin_y = msg.info.origin.position.x, msg.info.origin.position.y

    def get_cell_cost(self, x: float, y: float) -> int:
        """Fetch cost of a single point (used for robot's current position)"""
        if self.map_data is None: return 100
        raw_col = int((x - self.map_origin_x) / self.map_res)
        raw_row = int((y - self.map_origin_y) / self.map_res)

        MARGIN = 15
        if raw_col < MARGIN or raw_col >= (self.map_width - MARGIN) or \
           raw_row < MARGIN or raw_row >= (self.map_height - MARGIN):
            return 100

        col = max(0, min(raw_col, self.map_width - 1))
        row = max(0, min(raw_row, self.map_height - 1))
        cost = self.map_data[row * self.map_width + col]
        return 100 if cost == -1 else cost

    def get_cell_costs_vectorized(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Fetch costs for ALL 144 candidate points in one C-optimized step"""
        if self.map_data is None: return np.full(xs.shape, 100)
        
        raw_cols = ((xs - self.map_origin_x) / self.map_res).astype(np.int32)
        raw_rows = ((ys - self.map_origin_y) / self.map_res).astype(np.int32)
        
        MARGIN = 15
        invalid = (raw_cols < MARGIN) | (raw_cols >= (self.map_width - MARGIN)) | \
                  (raw_rows < MARGIN) | (raw_rows >= (self.map_height - MARGIN))
                  
        cols = np.clip(raw_cols, 0, self.map_width - 1)
        rows = np.clip(raw_rows, 0, self.map_height - 1)
        
        costs = self.map_data[rows * self.map_width + cols]
        costs = np.where((costs == -1) | invalid, 100, costs)
        return costs

    def is_path_clear(self, start_x: float, start_y: float, end_x: float, end_y: float, max_cost: int) -> bool:
        """Raycasts a single line to ensure no obstacles exist between robot and goal"""
        if self.map_data is None: return False
        dist = math.hypot(end_x - start_x, end_y - start_y)
        if dist < 0.05: return self.get_cell_cost(end_x, end_y) <= max_cost

        step_size = 0.10 if dist <= 0.8 else 0.05
        steps = max(2, int(dist / step_size))

        t = np.linspace(0.0, 1.0, steps)
        px = start_x + t * (end_x - start_x)
        py = start_y + t * (end_y - start_y)

        raw_cols = ((px - self.map_origin_x) / self.map_res).astype(np.int32)
        raw_rows = ((py - self.map_origin_y) / self.map_res).astype(np.int32)

        MARGIN = 15
        if np.any(raw_cols < MARGIN) or np.any(raw_cols >= (self.map_width - MARGIN)) or \
           np.any(raw_rows < MARGIN) or np.any(raw_rows >= (self.map_height - MARGIN)):
            return False

        cols = np.clip(raw_cols, 0, self.map_width - 1)
        rows = np.clip(raw_rows, 0, self.map_height - 1)
        indices = rows * self.map_width + cols
        costs = self.map_data[indices]

        return not (np.any(costs > max_cost) or np.any(costs == -1))

    def check_path_feasibility(self, start_x, start_y, end_x, end_y, max_cost):
        """Checks direct paths first, then falls back to 'dogleg' (V-shaped) paths if blocked"""
        if self.is_path_clear(start_x, start_y, end_x, end_y, max_cost):
            return True, 'STRAIGHT'
            
        direct_angle = math.atan2(end_y - start_y, end_x - start_x)
        mid_dist = math.hypot(end_x - start_x, end_y - start_y) / 2.0
        
        test_angles = np.deg2rad([30, -30, 45, -45, 60, -60, 90, -90])
        mid_x = start_x + (mid_dist * np.cos(direct_angle + test_angles))
        mid_y = start_y + (mid_dist * np.sin(direct_angle + test_angles))
        
        for i in range(len(test_angles)):
            if self.is_path_clear(start_x, start_y, mid_x[i], mid_y[i], max_cost):
                if self.is_path_clear(mid_x[i], mid_y[i], end_x, end_y, max_cost):
                    return True, 'DOGLEG'
        return False, 'NONE'

    def get_adaptive_weights(self, current_cost, dist_to_enemy):
        w_dist, w_speed, w_turn, w_cost = 12.0, 2.0, 3.0, 0.5
        if current_cost > 20: 
            w_cost, w_turn = 1.5, 0.5
        if dist_to_enemy < 1.2: 
            w_dist, w_speed = 22.0, 4.0
        return {'w_dist': w_dist, 'w_speed': w_speed, 'w_turn': w_turn, 'w_cost': w_cost}

    def evaluate_and_select_best_escape(self, our_x, our_y, our_yaw, enemy_x, enemy_y, enemy_yaw):
        """
        The core of the evasion calculator. Scores all valid points based on distance to enemy,
        front cone penalties, directional momentum, and wall repulsion. 
        Only runs raycasts on the Top 10 to save CPU.
        """
        c_x = our_x + self.cand_dx
        c_y = our_y + self.cand_dy
        c_r = self.cand_r
        
        c_cost = self.get_cell_costs_vectorized(c_x, c_y)
        current_cost = self.get_cell_cost(our_x, our_y)
        
        # Robustness: Cap dynamic threshold at 89 so paths are never projected through solid walls
        effective_max_cost = min(89, max(self.MAX_SAFE_COST, current_cost))
        
        valid_mask = c_cost <= effective_max_cost
        if not np.any(valid_mask): return None, -1000.0
            
        c_x, c_y, c_r, c_cost = c_x[valid_mask], c_y[valid_mask], c_r[valid_mask], c_cost[valid_mask]
        cand_angle = np.arctan2(c_y - our_y, c_x - our_x)

        weights = self.get_adaptive_weights(current_cost, math.hypot(our_x - enemy_x, our_y - enemy_y))
        dist_to_enemy = np.hypot(c_x - enemy_x, c_y - enemy_y)
        scores = (dist_to_enemy * weights['w_dist']) + (c_r * weights['w_speed'])

        away_dx = our_x - enemy_x
        away_dy = our_y - enemy_y
        away_angle = math.atan2(away_dy, away_dx)
        
        direct_away_blocked = self.get_cell_cost(our_x + math.cos(away_angle), our_y + math.sin(away_angle)) > 20
        tangent_left = away_angle + (math.pi / 2.0)
        tangent_right = away_angle - (math.pi / 2.0)

        # Slide logic if direct retreat is blocked
        if direct_away_blocked:
            diff_left = np.abs(np.arctan2(np.sin(cand_angle - tangent_left), np.cos(cand_angle - tangent_left)))
            diff_right = np.abs(np.arctan2(np.sin(cand_angle - tangent_right), np.cos(cand_angle - tangent_right)))
            scores += (np.pi - np.minimum(diff_left, diff_right)) * 12.0

            if self.slide_direction_bias == 'LEFT':
                scores = np.where(diff_left < diff_right, scores + 20.0, scores)
            elif self.slide_direction_bias == 'RIGHT':
                scores = np.where(diff_right < diff_left, scores + 20.0, scores)
        else:
            angle_diff_away = np.abs(np.arctan2(np.sin(cand_angle - away_angle), np.cos(cand_angle - away_angle)))
            scores += (np.pi - angle_diff_away) * 10.0

        # Front Cone Penalty: Don't dive across the enemy's nose
        angle_from_enemy_to_cand = np.arctan2(c_y - enemy_y, c_x - enemy_x)
        front_diff = np.abs(np.arctan2(np.sin(angle_from_enemy_to_cand - enemy_yaw), np.cos(angle_from_enemy_to_cand - enemy_yaw)))
        in_front_view = front_diff < np.deg2rad(60.0)
        getting_closer = dist_to_enemy < math.hypot(our_x - enemy_x, our_y - enemy_y)
        scores = np.where(in_front_view & getting_closer, scores - 50.0, scores)

        # Momentum: Stick to the current escape path to stop zigzagging
        if self.last_chosen_angle is not None:
            angle_diff_from_last = np.abs(np.arctan2(np.sin(cand_angle - self.last_chosen_angle), np.cos(cand_angle - self.last_chosen_angle)))
            scores -= (angle_diff_from_last * 25.0)

        # Wall Repulsion Magnet: Aggressively seek open space if currently touching a wall
        cost_increase = c_cost - current_cost
        scores = np.where(cost_increase > 0, scores - (cost_increase * 3.5), scores)
        scores = np.where(c_cost > 20, scores - (c_cost * 2.5), scores)
        
        if current_cost > 10:
            cost_decrease = current_cost - c_cost
            scores = np.where(cost_decrease > 0, scores + (cost_decrease * 4.0), scores)
        scores = np.where(c_cost == 0, scores + 35.0, scores)

        turn_penalty = np.abs(np.arctan2(np.sin(cand_angle - our_yaw), np.cos(cand_angle - our_yaw)))
        scores -= turn_penalty * weights['w_turn']
        scores -= c_cost * weights['w_cost']

        if self.active_target_x is not None and self.active_target_y is not None:
            active_dists = np.hypot(c_x - self.active_target_x, c_y - self.active_target_y)
            scores = np.where(active_dists < 0.7, scores + 40.0, scores)

        # TOP-K RAYCAST OPTIMIZATION
        top_k = min(10, len(scores))
        top_indices = np.argsort(scores)[-top_k:][::-1]

        best_point = None
        best_score = -float('inf')

        for idx in top_indices:
            tx, ty, t_angle, t_base_score = c_x[idx], c_y[idx], cand_angle[idx], scores[idx]
            
            is_clear, path_type = self.check_path_feasibility(our_x, our_y, tx, ty, effective_max_cost)
            if is_clear:
                final_score = t_base_score + (15.0 if path_type == 'STRAIGHT' else 0.0)
                if final_score > best_score:
                    best_score = final_score
                    best_point = (tx, ty, t_angle)

        if best_point is not None:
            self.active_target_x, self.active_target_y = best_point[0], best_point[1]
            self.last_chosen_angle = best_point[2]
            dl = abs(math.atan2(math.sin(best_point[2] - tangent_left), math.cos(best_point[2] - tangent_left)))
            dr = abs(math.atan2(math.sin(best_point[2] - tangent_right), math.cos(best_point[2] - tangent_right)))
            self.slide_direction_bias = 'LEFT' if dl < dr else 'RIGHT'
            
        return best_point, best_score

    def execute_emergency_escape(self, global_x: float, global_y: float, enemy_x: float, enemy_y: float):
        """Star-pattern fallback that ignores raycasting and just tries to slide down the cost gradient"""
        em_radii = np.array([0.3, 0.6, 0.9])
        angles = np.deg2rad(np.arange(0, 360, 15)) 
        
        r_grid, a_grid = np.meshgrid(em_radii, angles)
        cand_x = global_x + (r_grid * np.cos(a_grid)).flatten()
        cand_y = global_y + (r_grid * np.sin(a_grid)).flatten()
        
        costs = self.get_cell_costs_vectorized(cand_x, cand_y)
        valid_mask = costs < 90

        if np.any(valid_mask):
            valid_x, valid_y, valid_costs = cand_x[valid_mask], cand_y[valid_mask], costs[valid_mask]
            dist_to_enemy = np.hypot(valid_x - enemy_x, valid_y - enemy_y)
            # Maximize distance, heavily minimize cost
            scores = (dist_to_enemy * 25.0) - (valid_costs * 1.5)
            
            best_idx = np.argmax(scores)
            best_x, best_y = valid_x[best_idx], valid_y[best_idx]
            best_yaw = math.atan2(best_y - global_y, best_x - global_x)
            return best_x, best_y, best_yaw

        dx, dy = global_x - enemy_x, global_y - enemy_y
        escape_yaw = math.atan2(dy, dx) if math.hypot(dx, dy) > 0.01 else 0.0
        return global_x, global_y, escape_yaw

    def control_loop(self):
        now = self.get_clock().now().nanoseconds / 1e9

        if self.map_data is None: return

        if self.current_state != 'EVADING':
            self.last_target_x, self.last_target_y = None, None
            self.active_target_x, self.active_target_y = None, None
            self.slide_direction_bias, self.last_chosen_angle = None, None
            return

        try:
            tf_global = self.tf_buffer.lookup_transform(self.map_frame, self.robot_frame, rclpy.time.Time())
            tf_enemy_global = self.tf_buffer.lookup_transform(self.map_frame, self.enemy_frame, rclpy.time.Time())
            self.last_tf_time = now
        except tf2_ros.TransformException:
            # Dead Man's Switch: Hold position if sensors fail
            if now - self.last_tf_time > 0.5:
                if now - self.last_log_time > 1.0:
                    self.get_logger().warn("TF Data lost! Evasion halted to prevent runaway.")
                    self.last_log_time = now
                if self.last_target_x is not None:
                    t = TransformStamped()
                    t.header.stamp = self.get_clock().now().to_msg()
                    t.header.frame_id = self.map_frame
                    t.child_frame_id = self.target_frame
                    t.transform.translation.x = self.last_target_x
                    t.transform.translation.y = self.last_target_y
                    t.transform.rotation.w = 1.0
                    self.tf_broadcaster.sendTransform(t)
            return

        global_x, global_y = tf_global.transform.translation.x, tf_global.transform.translation.y
        our_yaw = self.get_yaw_from_quaternion(tf_global.transform.rotation)
        enemy_global_x, enemy_global_y = tf_enemy_global.transform.translation.x, tf_enemy_global.transform.translation.y
        enemy_global_yaw = self.get_yaw_from_quaternion(tf_enemy_global.transform.rotation)

        best_target, max_score = self.evaluate_and_select_best_escape(
            global_x, global_y, our_yaw, enemy_global_x, enemy_global_y, enemy_global_yaw
        )

        # Trigger Emergency if standard paths are blocked OR if scores are deeply negative (wall corner)
        if best_target is None or max_score < -25.0:
            if now - self.last_log_time > 1.0:
                self.get_logger().warn("Standard paths rejected/blocked. Executing EMERGENCY ESCAPE.")
                self.last_log_time = now
            target_x, target_y, target_yaw = self.execute_emergency_escape(global_x, global_y, enemy_global_x, enemy_global_y)
        else:
            target_x, target_y, target_yaw = best_target

        if self.last_target_x is not None and self.last_target_y is not None:
            alpha = 0.5
            target_x = alpha * target_x + (1.0 - alpha) * self.last_target_x
            target_y = alpha * target_y + (1.0 - alpha) * self.last_target_y
            if math.hypot(target_x - self.last_target_x, target_y - self.last_target_y) < self.MIN_UPDATE_DIST: 
                return        
        
        self.last_target_x, self.last_target_y = target_x, target_y

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.target_frame
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = target_x, target_y, 0.0
        t.transform.rotation.z, t.transform.rotation.w = math.sin(target_yaw / 2.0), math.cos(target_yaw / 2.0)
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(EvasionCalculator())
    rclpy.shutdown()

if __name__ == '__main__':
    main()