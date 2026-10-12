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
        # Extended search radii [0.6, 1.0, 1.5, 2.2] for quick evasive turns and long-range sprint vectors.
        self.sample_radii = [0.6, 1.0, 1.5, 2.2]
        
        # TUNING POINT: Search angle resolution. 10 degrees (36 steps) balances spatial coverage and CPU raycast speed.
        self.angle_steps = [(math.cos(math.radians(deg)), math.sin(math.radians(deg))) for deg in range(0, 360, 10)]

        self.map_data = None
        self.map_res = 0.05
        self.map_width = 0
        self.map_height = 0
        self.map_origin_x = 0.0 
        self.map_origin_y = 0.0
        
        # TUNING POINT: Cost threshold for evasion paths (OccupancyGrid scale 0-100).
        # Set to 40 so escape routes stay well clear of wall inflation zones.
        self.MAX_SAFE_COST = 40

        self.last_target_x, self.last_target_y = None, None
        
        # TUNING POINT: How much the target must move before broadcasting a new evasion point.
        self.MIN_UPDATE_DIST = 0.02
        self.memory_bank: List[Tuple[float, float, float]] = []
        self.memory_space = 5

        # ENHANCEMENT: Enemy velocity tracking for predictive trajectory avoidance
        self.last_enemy_x = None
        self.last_enemy_y = None
        self.last_enemy_time = None
        self.enemy_vx = 0.0
        self.enemy_vy = 0.0

        # ENHANCEMENT: Active target hysteresis to prevent path chatter while keeping search dynamic
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
        # This prevents target TFs from ever landing near the edge of the map
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
        # Getting the distance between the our bot and the select pose
        dist = math.hypot(end_x - start_x, end_y - start_y)

        # Dynamic step size based on the distance
        step_size = 0.10 if dist <= 0.8 else 0.05
        # Find the number of mini goals / steps needed
        steps = max(2, int(dist / step_size))

        # Create those mini goals in the range of 0.0 to 1.0 and then scale them to the actual path
        t = np.linspace(0.0, 1.0, steps)
        # stores distance between the start and end point spaced out by step_size
        px = start_x + t * (end_x - start_x)
        py = start_y + t * (end_y - start_y)

        # Ensure rays checking outside map bounds return invalid path
        raw_cols = ((px - self.map_origin_x) / self.map_res).astype(np.int32)
        raw_rows = ((py - self.map_origin_y) / self.map_res).astype(np.int32)
        if np.any(raw_cols < 0) or np.any(raw_cols >= self.map_width) or \
           np.any(raw_rows < 0) or np.any(raw_rows >= self.map_height):
            return False

        cols = np.clip(raw_cols, 0, self.map_width - 1)
        rows = np.clip(raw_rows, 0, self.map_height - 1)

        # indices is an array it is stored as np_array
        indices = rows * self.map_width + cols
        # costs is an array of costs
        costs = self.map_data[indices]

        # If any cost above our danger threshold or is unknown (-1), return False
        if np.any(costs > self.MAX_SAFE_COST) or np.any(costs == -1): return False
        return True

    def is_path_clear_dogleg(self, start_x, start_y, end_x, end_y) -> bool:
        # 1. Fast Pass: Straight line
        if self.is_path_clear(start_x, start_y, end_x, end_y):
            return True
            
        direct_angle = math.atan2(end_y - start_y, end_x - start_x)
        total_dist = math.hypot(end_x - start_x, end_y - start_y)
        mid_dist = total_dist / 2.0
        
        # 2. Bending Pass: Try 45 deg first (tighter), then 65 deg (wider)
        test_angles = [
            math.radians(45), math.radians(-45),
            math.radians(65), math.radians(-65)
        ]
        
        for offset in test_angles:
            mid_x = start_x + (mid_dist * math.cos(direct_angle + offset))
            mid_y = start_y + (mid_dist * math.sin(direct_angle + offset))

            # Checking if elbow path is clear
            # If we can reach the elbow, and the elbow can reach the target
            if self.is_path_clear(start_x, start_y, mid_x, mid_y):
                if self.is_path_clear(mid_x, mid_y, end_x, end_y):
                    return True
                    
        return False

    def check_memory_bank(self, our_x, our_y, enemy_x, enemy_y):

        # Run forward if memory bank is is not empty
        if not self.memory_bank: return []

        # temporary list to hold valid points
        still_valid_points = []
        current_dist_to_enemy = math.hypot(our_x - enemy_x, our_y - enemy_y)
        for mx, my, myaw in self.memory_bank:
            mem_dist_to_enemy = math.hypot(mx - enemy_x, my - enemy_y)
            # TUNING POINT: Hysteresis for keeping the current escape point.
            if mem_dist_to_enemy <= (current_dist_to_enemy + 0.3): continue
            if not self.is_path_clear_dogleg(our_x, our_y, mx, my): continue
            if self.get_cell_cost(mx, my) > self.MAX_SAFE_COST: continue
            still_valid_points.append((mx, my, myaw))
        return still_valid_points

    # Weight provider
    def get_adaptive_weights(self, current_cost, dist_to_enemy):
        # TUNING POINT: These weights dictate the robot's "personality" when running away.
        # w_dist = Prioritize getting far from enemy
        # w_speed = Prioritize longer escape vectors
        # w_turn = Penalty for having to turn around to escape
        # w_cost = Penalty for running near walls
        w_dist, w_speed, w_turn, w_cost = 15.0, 2.5, 2.5, 0.4
        
        # TUNING POINT: Emergency weights if near a wall or enemy
        # Dynamic weight adjuster
        bad_cost = 20
        min_dist_to_enemy = 1.2
        if current_cost > bad_cost: w_cost, w_turn = 1.2, 0.5
        if dist_to_enemy < min_dist_to_enemy: w_dist, w_speed = 25.0, 5.0
        return {'w_dist': w_dist, 'w_speed': w_speed, 'w_turn': w_turn, 'w_cost': w_cost}

    # ENHANCEMENT: Enemy velocity filtering for dynamic trajectory prediction
    def update_enemy_velocity(self, enemy_x: float, enemy_y: float, current_time: float):
        if self.last_enemy_x is not None and self.last_enemy_time is not None:
            dt = current_time - self.last_enemy_time
            if 0.01 < dt < 1.0:
                raw_vx = (enemy_x - self.last_enemy_x) / dt
                raw_vy = (enemy_y - self.last_enemy_y) / dt
                # Low-pass filter to smooth out transform jitter
                self.enemy_vx = 0.6 * self.enemy_vx + 0.4 * raw_vx
                self.enemy_vy = 0.6 * self.enemy_vy + 0.4 * raw_vy
        self.last_enemy_x = enemy_x
        self.last_enemy_y = enemy_y
        self.last_enemy_time = current_time

    # Checking if future path is also good
    def evaluate_future_options(self, cand_x, cand_y):

        future_clear_branches, blocked_branches = 0, 0
        MAX_ALLOWED_BLOCKED = 4
        # the distance the path is projected
        probe_radius = self.sample_radii[0]

        # step by 4 (speed things up) for a particular path
        for cos_val, sin_val in self.angle_steps[::4]:
            next_x = cand_x + (probe_radius * cos_val)
            next_y = cand_y + (probe_radius * sin_val)
            # Check if this path is clear
            if self.is_path_clear_dogleg(cand_x, cand_y, next_x, next_y) and self.get_cell_cost(next_x, next_y) <= self.MAX_SAFE_COST:
                # if a lot of 
                future_clear_branches += 1
            else:
                blocked_branches += 1
                # So if more then MAX_ALLOWED_BLOCKED are present, we stop
                if blocked_branches >= MAX_ALLOWED_BLOCKED: return 0.0
        return future_clear_branches * 1.5

    def get_safe_escape_points(self, robot_x, robot_y):
        safe_points = []
        for r in self.sample_radii:
            for cos_val, sin_val in self.angle_steps:
                cand_x = robot_x + (r * cos_val)
                cand_y = robot_y + (r * sin_val)
                # Check if the path is clear
                if self.is_path_clear_dogleg(robot_x, robot_y, cand_x, cand_y):
                    # Add to safe points with cost for scoring later
                    safe_points.append((cand_x, cand_y, r, self.get_cell_cost(cand_x, cand_y)))
        return safe_points

    def get_best_escape_target(self, safe_points, our_x, our_y, our_yaw, enemy_x, enemy_y):
        # Setting variables
        # best_points
        # higest_score to - infinit at start
        # setting an array to store all points
        best_point, highest_score, all_scored_points = None, -float('inf'), []

        # weight values
        weights = self.get_adaptive_weights(self.get_cell_cost(our_x, our_y), math.hypot(our_x - enemy_x, our_y - enemy_y))

        # ENHANCEMENT: Predict enemy position 1.0s into future based on velocity vector
        pred_time = 1.0
        pred_enemy_x = enemy_x + (self.enemy_vx * pred_time)
        pred_enemy_y = enemy_y + (self.enemy_vy * pred_time)

        # Run through the safe points collect from the get_safe_escape_point method
        for cand_x, cand_y, r, cost in safe_points:
            # Score distance against both current AND predicted enemy position to prevent running into their path
            dist_curr = math.hypot(cand_x - enemy_x, cand_y - enemy_y)
            dist_pred = math.hypot(cand_x - pred_enemy_x, cand_y - pred_enemy_y)
            effective_enemy_dist = (0.4 * dist_curr) + (0.6 * dist_pred)

            # Add points to distance and radius usage
            score = (effective_enemy_dist * weights['w_dist']) + (r * weights['w_speed'])

            # Get the angle difference between us the goal pose
            # deduct points when there big angle difference between us the goal pose
            angle_to_cand = math.atan2(cand_y - our_y, cand_x - our_x)
            score -= abs(math.atan2(math.sin(angle_to_cand - our_yaw), math.cos(angle_to_cand - our_yaw))) * weights['w_turn']
            score -= cost * weights['w_cost']
            # Add extra points when the cost is very low
            if 0 <= cost < 10: score += 12.0

            # ENHANCEMENT: Active target hysteresis bonus (+15.0) to maintain path commitment without absolute lock
            if self.active_target_x is not None and self.active_target_y is not None:
                if math.hypot(cand_x - self.active_target_x, cand_y - self.active_target_y) < 0.4:
                    score += 15.0

            # if the future also look good, give that path extra points
            score += self.evaluate_future_options(cand_x, cand_y)
            all_scored_points.append((score, cand_x, cand_y, angle_to_cand))

            # Storing the best_point and higest_score values
            if score > highest_score: 
                highest_score = score 
                best_point = (cand_x, cand_y, angle_to_cand)

        # Sorting Alg, based on the score
        all_scored_points.sort(key=lambda item: item[0], reverse=True)

        # Store the best first #memory space paths
        # store only the x, y, rotation values only
        self.memory_bank = [(item[1], item[2], item[3]) for item in all_scored_points[:self.memory_space]]
        if best_point is not None:
            self.active_target_x, self.active_target_y = best_point[0], best_point[1]

        return best_point

    # ENHANCEMENT: Wall-aware radial sweep fallback when trapped or no full dogleg paths are free
    def execute_emergency_escape(self, global_x: float, global_y: float, enemy_x: float, enemy_y: float, our_yaw: float):
        best_x, best_y, best_yaw, max_score = None, None, None, -float('inf')

        # Radial sweep around our bot from 0.4m outward to find the longest open-space vector
        for dist in [0.4, 0.7, 1.0, 1.3]:
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

        # Ultimate fallback if backed directly against a corner
        dx = global_x - enemy_x
        dy = global_y - enemy_y
        dist = math.hypot(dx, dy)
        if dist == 0: dist = 0.01
        return global_x + (dx / dist) * 0.5, global_y + (dy / dist) * 0.5, math.atan2(dy, dx)

    def control_loop(self):
        if self.map_data is None:
            return

        if self.current_state != 'EVADING':
            self.last_target_x = None
            self.last_target_y = None
            self.active_target_x = None
            self.active_target_y = None
            self.memory_bank.clear()
            return

        # Getting our position
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

        now_sec = self.get_clock().now().nanoseconds / 1e9
        self.update_enemy_velocity(enemy_global_x, enemy_global_y, now_sec)

        safe_points = self.get_safe_escape_points(global_x, global_y)
        
        if not safe_points:
            # Wall-aware radial sweep fallback
            self.get_logger().warn("No standard safe points! Executing radial sweep escape.")
            target_x, target_y, target_yaw = self.execute_emergency_escape(global_x, global_y, enemy_global_x, enemy_global_y, our_yaw)
        else:
            # get the best path and put that as the goal
            best_target = self.get_best_escape_target(safe_points, global_x, global_y, our_yaw, enemy_global_x, enemy_global_y)
            if best_target is None: 
                return
            target_x, target_y, target_yaw = best_target

        # ENHANCEMENT: Low-pass filter (EMA) target pose to eliminate high-frequency dynamic frame jitter
        if self.last_target_x is not None and self.last_target_y is not None:
            alpha = 0.35  # Smoothing factor
            target_x = alpha * target_x + (1.0 - alpha) * self.last_target_x
            target_y = alpha * target_y + (1.0 - alpha) * self.last_target_y

        if self.last_target_x is not None: 
            if math.hypot(target_x - self.last_target_x, target_y - self.last_target_y) < self.MIN_UPDATE_DIST: return        
        
        self.last_target_x, self.last_target_y = target_x, target_y

        # UNCONDITIONAL BROADCAST: 
        # Ensures goal_sender always has a target to follow
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