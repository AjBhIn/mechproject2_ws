#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
import tf2_ros
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import String
from typing import List, Tuple
import numpy as np


class EvasionCalculator(Node):
    def __init__(self):
        super().__init__('evasion_calculator')

        self.map_frame = 'map'
        self.robot_frame = 'our_bot/base_link'
        self.enemy_frame = 'enemy_bot/base_link'
        self.target_frame = 'dynamic_nav_target'
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.current_state = 'CHASING'
        # Topic run continuously
        self.state_sub = self.create_subscription(String, 'our_bot/robot_state', self.state_cb, 10)
        self.costmap_sub = self.create_subscription(OccupancyGrid, '/our_bot/global_costmap/costmap', self.costmap_callback, 10)

        # TUNING POINT: Distances (meters) to search for safe escape points.
        self.sample_radii = [0.6, 1.0, 1.5, 2.0]
        
        # TUNING POINT: Search angle resolution. 10 degrees (36 steps) balances coverage and CPU speed.
        self.angle_steps = [(math.cos(math.radians(deg)), math.sin(math.radians(deg))) for deg in range(0, 360, 10)]

        self.map_data = None
        self.map_res = 0.05
        self.map_width = 0
        self.map_height = 0
        self.map_origin_x = 0.0 
        self.map_origin_y = 0.0
        
        # TUNING POINT: Cost threshold for evasion paths (OccupancyGrid scale 0-100).
        self.MAX_SAFE_COST = 40

        self.last_target_x, self.last_target_y = None, None
        
        # Active target lock to maintain path commitment and stop zigzagging
        self.active_target_x = None
        self.active_target_y = None
        self.slide_direction_bias = None  # Persistent slide bias (+1 or -1) when wall-sliding

        # Smooth goal selector tracking
        self.last_goal_angle = None

        # The runner!
        self.timer = self.create_timer(0.1, self.control_loop)

    # getting the state
    def state_cb(self, msg):
        self.current_state = msg.data

    # angle to quaternion
    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    # Get the map info from the topic
    def costmap_callback(self, msg: OccupancyGrid):

        self.map_data = np.array(msg.data, dtype=np.int16)
        self.map_res = msg.info.resolution

        # Added the number as way to bring the cells in from the edge of the map
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y

    # Getting cost of ONE CELL AT A TIME
    def get_cell_cost(self, x: float, y: float) -> int:

        if self.map_data is None: return 100
        raw_col = int((x - self.map_origin_x) / self.map_res)
        raw_row = int((y - self.map_origin_y) / self.map_res)

        # VIRTUAL BOUNDARY PADDING: 
        # Treat the outer 15 cells (0.75 meters at 0.05m res) as solid walls
        MARGIN = 15
        if raw_col < MARGIN or raw_col >= (self.map_width - MARGIN) or \
           raw_row < MARGIN or raw_row >= (self.map_height - MARGIN):
            return 100

        col = max(0, min(raw_col, self.map_width - 1))
        row = max(0, min(raw_row, self.map_height - 1))

        cost = self.map_data[row * self.map_width + col]
        return 100 if cost == -1 else cost

    # Getting Cost of ONE Line AT A TIME
    def is_path_clear(self, start_x: float, start_y: float, end_x: float, end_y: float) -> bool:

        if self.map_data is None: return False
        dist = math.hypot(end_x - start_x, end_y - start_y)

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

        if np.any(costs > self.MAX_SAFE_COST) or np.any(costs == -1): return False
        return True

    def check_path_feasibility(self, start_x, start_y, end_x, end_y):
        if self.is_path_clear(start_x, start_y, end_x, end_y):
            return True, 'STRAIGHT'
            
        direct_angle = math.atan2(end_y - start_y, end_x - start_x)
        total_dist = math.hypot(end_x - start_x, end_y - start_y)
        mid_dist = total_dist / 2.0
        
        test_angles = [math.radians(45), math.radians(-45)]
        
        for offset in test_angles:
            mid_x = start_x + (mid_dist * math.cos(direct_angle + offset))
            mid_y = start_y + (mid_dist * math.sin(direct_angle + offset))

            if self.is_path_clear(start_x, start_y, mid_x, mid_y):
                if self.is_path_clear(mid_x, mid_y, end_x, end_y):
                    return True, 'DOGLEG'
                    
        return False, 'NONE'

    def get_adaptive_weights(self, current_cost, dist_to_enemy):
        w_dist, w_speed, w_turn, w_cost = 12.0, 2.0, 3.0, 0.5
        bad_cost = 20
        min_dist_to_enemy = 1.2
        if current_cost > bad_cost: w_cost, w_turn = 1.5, 0.5
        if dist_to_enemy < min_dist_to_enemy: w_dist, w_speed = 22.0, 4.0
        return {'w_dist': w_dist, 'w_speed': w_speed, 'w_turn': w_turn, 'w_cost': w_cost}

    def get_safe_escape_points(self, robot_x, robot_y):
        safe_points = []
        for r in self.sample_radii:
            for cos_val, sin_val in self.angle_steps:
                cand_x = robot_x + (r * cos_val)
                cand_y = robot_y + (r * sin_val)

                if self.get_cell_cost(cand_x, cand_y) > self.MAX_SAFE_COST:
                    continue

                is_clear, path_type = self.check_path_feasibility(robot_x, robot_y, cand_x, cand_y)
                if is_clear:
                    safe_points.append((cand_x, cand_y, r, self.get_cell_cost(cand_x, cand_y), path_type))
        return safe_points

    def get_best_escape_target(self, safe_points, our_x, our_y, our_yaw, enemy_x, enemy_y):
        best_point, highest_score = None, -float('inf')
        current_cost = self.get_cell_cost(our_x, our_y)
        weights = self.get_adaptive_weights(current_cost, math.hypot(our_x - enemy_x, our_y - enemy_y))

        # Vector pointing directly away from enemy
        away_dx = our_x - enemy_x
        away_dy = our_y - enemy_y
        away_angle = math.atan2(away_dy, away_dx)

        # Check if the direct 'away' path leads into a wall (inflation > 20 within 1.0m)
        direct_away_x = our_x + 1.0 * math.cos(away_angle)
        direct_away_y = our_y + 1.0 * math.sin(away_angle)
        direct_away_blocked = self.get_cell_cost(direct_away_x, direct_away_y) > 20

        # Calculate Tangential Vectors (slide left or slide right along the wall)
        tangent_left = away_angle + (math.pi / 2.0)
        tangent_right = away_angle - (math.pi / 2.0)

        for cand_x, cand_y, r, cost, path_type in safe_points:
            dist_to_enemy = math.hypot(cand_x - enemy_x, cand_y - enemy_y)
            score = (dist_to_enemy * weights['w_dist']) + (r * weights['w_speed'])

            cand_angle = math.atan2(cand_y - our_y, cand_x - our_x)

            # 1. TANGENTIAL DEFLECTION: If direct path away is blocked by a wall, reward sliding parallel to wall
            if direct_away_blocked:
                diff_left = abs(math.atan2(math.sin(cand_angle - tangent_left), math.cos(cand_angle - tangent_left)))
                diff_right = abs(math.atan2(math.sin(cand_angle - tangent_right), math.cos(cand_angle - tangent_right)))
                
                # Favor tangential sliding score
                slide_score = (math.pi - min(diff_left, diff_right)) * 12.0
                score += slide_score

                # Maintain slide direction hysteresis (+20.0) to stop left-right zigzagging along the wall
                if self.slide_direction_bias == 'LEFT' and diff_left < diff_right:
                    score += 20.0
                elif self.slide_direction_bias == 'RIGHT' and diff_right < diff_left:
                    score += 20.0
            else:
                # Direct away path is open—reward moving straight away
                angle_diff_away = abs(math.atan2(math.sin(cand_angle - away_angle), math.cos(cand_angle - away_angle)))
                score += (math.pi - angle_diff_away) * 10.0

            # 2. INFLATION SLOPE PENALTY: Severely penalize moving deeper into wall inflation!
            cost_increase = cost - current_cost
            if cost_increase > 0:
                score -= cost_increase * 3.5  # Heavy penalty for heading towards higher cost/walls

            turn_penalty = abs(math.atan2(math.sin(cand_angle - our_yaw), math.cos(cand_angle - our_yaw)))
            score -= turn_penalty * weights['w_turn']
            score -= cost * weights['w_cost']

            if path_type == 'STRAIGHT':
                score += 15.0

            if 0 <= cost < 10: 
                score += 10.0

            # Active target hysteresis (+25.0) locks the target point so DWB doesn't wobble
            if self.active_target_x is not None and self.active_target_y is not None:
                if math.hypot(cand_x - self.active_target_x, cand_y - self.active_target_y) < 0.4:
                    score += 25.0

            if score > highest_score: 
                highest_score = score 
                best_point = (cand_x, cand_y, cand_angle)

        if best_point is not None:
            self.active_target_x, self.active_target_y = best_point[0], best_point[1]
            
            # Remember whether we are sliding left or right to keep momentum smooth
            cand_angle = best_point[2]
            diff_left = abs(math.atan2(math.sin(cand_angle - tangent_left), math.cos(cand_angle - tangent_left)))
            diff_right = abs(math.atan2(math.sin(cand_angle - tangent_right), math.cos(cand_angle - tangent_right)))
            self.slide_direction_bias = 'LEFT' if diff_left < diff_right else 'RIGHT'

        return best_point

    def execute_emergency_escape(self, global_x: float, global_y: float, enemy_x: float, enemy_y: float):
        best_x, best_y, best_yaw, max_score = None, None, None, -float('inf')

        for dist in [0.4, 0.7, 1.0]:
            for cos_val, sin_val in self.angle_steps:
                cand_x = global_x + (dist * cos_val)
                cand_y = global_y + (dist * sin_val)
                if self.get_cell_cost(cand_x, cand_y) <= self.MAX_SAFE_COST:
                    if self.is_path_clear(global_x, global_y, cand_x, cand_y):
                        enemy_dist = math.hypot(cand_x - enemy_x, cand_y - enemy_y)
                        ray_score = (dist * 2.0) + (enemy_dist * 5.0)
                        if ray_score > max_score:
                            max_score = ray_score
                            best_x, best_y = cand_x, cand_y
                            best_yaw = math.atan2(sin_val, cos_val)

        if best_x is not None:
            return best_x, best_y, best_yaw

        dx = global_x - enemy_x
        dy = global_y - enemy_y
        dist = math.hypot(dx, dy)
        if dist == 0: dist = 0.01
        return global_x + (dx / dist) * 0.4, global_y + (dy / dist) * 0.4, math.atan2(dy, dx)

    # Angular Rate Limiter: slews direction gradually to stop DWB jitter, uncapping for emergency breaks
    def apply_angular_goal_slewing(self, robot_x: float, robot_y: float, target_x: float, target_y: float, dist_to_enemy: float):
        desired_angle = math.atan2(target_y - robot_y, target_x - robot_x)

        if self.last_goal_angle is None:
            self.last_goal_angle = desired_angle
            return target_x, target_y, desired_angle

        angle_diff = math.atan2(
            math.sin(desired_angle - self.last_goal_angle),
            math.cos(desired_angle - self.last_goal_angle)
        )

        # Dynamic Slew Rate:
        # Under 1.2m: permit instant 180 deg snap for emergency escape
        # Over 1.2m: smoothly limit change to 25 deg per 0.1s tick
        if dist_to_enemy < 1.2:
            max_step = math.radians(180.0)
        else:
            max_step = math.radians(25.0)

        clamped_diff = max(-max_step, min(max_step, angle_diff))
        slewed_angle = self.last_goal_angle + clamped_diff

        radius = math.hypot(target_x - robot_x, target_y - robot_y)
        slewed_x = robot_x + radius * math.cos(slewed_angle)
        slewed_y = robot_y + radius * math.sin(slewed_angle)

        # If slewed coordinate lands in an obstacle, fall back to safe target directly
        if self.get_cell_cost(slewed_x, slewed_y) > self.MAX_SAFE_COST or not self.is_path_clear(robot_x, robot_y, slewed_x, slewed_y):
            self.last_goal_angle = desired_angle
            return target_x, target_y, desired_angle

        self.last_goal_angle = slewed_angle
        return slewed_x, slewed_y, slewed_angle

    def control_loop(self):
        if self.map_data is None or self.current_state != 'EVADING':
            self.last_target_x = None
            self.last_target_y = None
            self.active_target_x = None
            self.active_target_y = None
            self.slide_direction_bias = None
            self.last_goal_angle = None
            return

        try:
            tf_global = self.tf_buffer.lookup_transform(
                self.map_frame, self.robot_frame, rclpy.time.Time(), 
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
            tf_enemy_global = self.tf_buffer.lookup_transform(
                self.map_frame, self.enemy_frame, rclpy.time.Time(), 
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
        except tf2_ros.TransformException:
            return

        global_x, global_y = tf_global.transform.translation.x, tf_global.transform.translation.y
        our_yaw = self.get_yaw_from_quaternion(tf_global.transform.rotation)
        enemy_global_x, enemy_global_y = tf_enemy_global.transform.translation.x, tf_enemy_global.transform.translation.y
        dist_to_enemy = math.hypot(global_x - enemy_global_x, global_y - enemy_global_y)

        safe_points = self.get_safe_escape_points(global_x, global_y)
        
        if not safe_points:
            target_x, target_y, target_yaw = self.execute_emergency_escape(global_x, global_y, enemy_global_x, enemy_global_y)
        else:
            best_target = self.get_best_escape_target(safe_points, global_x, global_y, our_yaw, enemy_global_x, enemy_global_y)
            if best_target is None: 
                return
            target_x, target_y, target_yaw = best_target

        # Smooth goal selector layer: limits drastic angle flips while preserving close evasion
        target_x, target_y, target_yaw = self.apply_angular_goal_slewing(
            global_x, global_y, target_x, target_y, dist_to_enemy
        )

        if self.last_target_x is not None and self.last_target_y is not None:
            alpha = 0.5
            target_x = alpha * target_x + (1.0 - alpha) * self.last_target_x
            target_y = alpha * target_y + (1.0 - alpha) * self.last_target_y

        self.last_target_x, self.last_target_y = target_x, target_y

        # Broadcast transform every cycle with current timestamp to ensure no goal timeouts
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