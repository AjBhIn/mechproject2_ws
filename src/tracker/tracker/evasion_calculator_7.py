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
        # Balanced radii for quick escapes and longer getaway vectors.
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
        
        # TUNING POINT: How much the target must move before broadcasting a new evasion point.
        self.MIN_UPDATE_DIST = 0.02
        self.memory_bank: List[Tuple[float, float, float]] = []
        self.memory_space = 5

        # Active target lock to maintain path commitment
        self.active_target_x = None
        self.active_target_y = None

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
        # This strictly prevents target TFs from ever landing near or outside the edge of the map
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

        # Strict map boundary check along the raycast line
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

    # Evaluates straight paths first, falling back to dogleg bending paths if direct line is blocked
    def check_path_feasibility(self, start_x, start_y, end_x, end_y):
        # 1. Straight path check (Fastest & Strongly Preferred)
        if self.is_path_clear(start_x, start_y, end_x, end_y):
            return True, 'STRAIGHT'
            
        # 2. Bending Pass (Fallback around obstacles)
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

    # Weight provider
    def get_adaptive_weights(self, current_cost, dist_to_enemy):
        w_dist, w_speed, w_turn, w_cost = 12.0, 2.0, 3.0, 0.4
        bad_cost = 20
        min_dist_to_enemy = 1.2
        if current_cost > bad_cost: w_cost, w_turn = 1.2, 0.5
        if dist_to_enemy < min_dist_to_enemy: w_dist, w_speed = 22.0, 4.0
        return {'w_dist': w_dist, 'w_speed': w_speed, 'w_turn': w_turn, 'w_cost': w_cost}

    def get_safe_escape_points(self, robot_x, robot_y):
        safe_points = []
        for r in self.sample_radii:
            for cos_val, sin_val in self.angle_steps:
                cand_x = robot_x + (r * cos_val)
                cand_y = robot_y + (r * sin_val)

                # Strict boundary and cost check on candidate point
                if self.get_cell_cost(cand_x, cand_y) > self.MAX_SAFE_COST:
                    continue

                is_clear, path_type = self.check_path_feasibility(robot_x, robot_y, cand_x, cand_y)
                if is_clear:
                    safe_points.append((cand_x, cand_y, r, self.get_cell_cost(cand_x, cand_y), path_type))
        return safe_points

    def get_best_escape_target(self, safe_points, our_x, our_y, our_yaw, enemy_x, enemy_y):
        best_point, highest_score = None, -float('inf')
        weights = self.get_adaptive_weights(self.get_cell_cost(our_x, our_y), math.hypot(our_x - enemy_x, our_y - enemy_y))

        # Vector pointing directly away from enemy
        away_dx = our_x - enemy_x
        away_dy = our_y - enemy_y
        away_angle = math.atan2(away_dy, away_dx)

        # Baseline distances along X and Y axes to check axis convergence
        curr_dx = abs(our_x - enemy_x)
        curr_dy = abs(our_y - enemy_y)

        for cand_x, cand_y, r, cost, path_type in safe_points:
            dist_to_enemy = math.hypot(cand_x - enemy_x, cand_y - enemy_y)
            score = (dist_to_enemy * weights['w_dist']) + (r * weights['w_speed'])

            cand_angle = math.atan2(cand_y - our_y, cand_x - our_x)
            
            # Align getaway vector directly away from enemy
            angle_diff_away = abs(math.atan2(math.sin(cand_angle - away_angle), math.cos(cand_angle - away_angle)))
            score += (math.pi - angle_diff_away) * 8.0

            # SOFT WEIGHT: Component-wise distance check (X/Y approach penalty)
            cand_dx = abs(cand_x - enemy_x)
            cand_dy = abs(cand_y - enemy_y)
            approaching_x = cand_dx < curr_dx
            approaching_y = cand_dy < curr_dy

            if approaching_x and approaching_y:
                score -= 35.0  # Penalize closing in on enemy along both axes
            elif approaching_x or approaching_y:
                score -= 10.0  # Minor penalty for closing in along one axis
            else:
                score += 15.0  # Reward opening up distance on both X and Y axes

            turn_penalty = abs(math.atan2(math.sin(cand_angle - our_yaw), math.cos(cand_angle - our_yaw)))
            score -= turn_penalty * weights['w_turn']
            score -= cost * weights['w_cost']

            # KEY PREFERENCE: Big score bonus (+18.0) for pure straight paths
            if path_type == 'STRAIGHT':
                score += 18.0

            if 0 <= cost < 10: 
                score += 8.0

            # Target hysteresis (+12.0) keeps robot committed to a path without flickering
            if self.active_target_x is not None and self.active_target_y is not None:
                if math.hypot(cand_x - self.active_target_x, cand_y - self.active_target_y) < 0.4:
                    score += 12.0

            if score > highest_score: 
                highest_score = score 
                best_point = (cand_x, cand_y, cand_angle)

        if best_point is not None:
            self.active_target_x, self.active_target_y = best_point[0], best_point[1]

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

    def control_loop(self):
        if self.map_data is None or self.current_state != 'EVADING':
            self.last_target_x = None
            self.last_target_y = None
            self.active_target_x = None
            self.active_target_y = None
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

        safe_points = self.get_safe_escape_points(global_x, global_y)
        
        if not safe_points:
            target_x, target_y, target_yaw = self.execute_emergency_escape(global_x, global_y, enemy_global_x, enemy_global_y)
        else:
            best_target = self.get_best_escape_target(safe_points, global_x, global_y, our_yaw, enemy_global_x, enemy_global_y)
            if best_target is None: 
                return
            target_x, target_y, target_yaw = best_target

        if self.last_target_x is not None and self.last_target_y is not None:
            alpha = 0.4
            target_x = alpha * target_x + (1.0 - alpha) * self.last_target_x
            target_y = alpha * target_y + (1.0 - alpha) * self.last_target_y

        if self.last_target_x is not None: 
            if math.hypot(target_x - self.last_target_x, target_y - self.last_target_y) < self.MIN_UPDATE_DIST: return        
        
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