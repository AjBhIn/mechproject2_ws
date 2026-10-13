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
        self.create_subscription(String, 'our_bot/robot_state', self.state_cb, 10)
        self.create_subscription(OccupancyGrid, '/our_bot/global_costmap/costmap', self.costmap_callback, 10)

        # TUNING: Deep radii so DWB maintains full sprint velocity rather than terminal braking
        self.sample_radii = [0.8, 1.4, 2.0]
        # 24 discrete angles (every 15 deg) balances high spatial coverage with low CPU load
        self.angle_steps = [(math.cos(math.radians(deg)), math.sin(math.radians(deg)), math.radians(deg)) 
                            for deg in range(0, 360, 15)]

        self.map_data = None
        self.map_res = 0.05
        self.map_width = 0
        self.map_height = 0
        self.map_origin_x = 0.0 
        self.map_origin_y = 0.0

        # Permissive threshold: corridors and pillar gaps (cost 20-40) are valid escape routes
        self.MAX_SAFE_COST = 45

        self.last_target_x, self.last_target_y = None, None
        self.last_away_angle = None
        self.last_goal_angle = None

        self.active_target_x = None
        self.active_target_y = None
        self.slide_direction_bias = None

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Tactical Evasion Calculator Initialized (Deep Horizon & Fast Tangents).")

    def state_cb(self, msg):
        self.current_state = msg.data

    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def costmap_callback(self, msg: OccupancyGrid):
        self.map_data = np.array(msg.data, dtype=np.int16)
        self.map_res = msg.info.resolution
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.map_origin_x = msg.info.origin.position.x
        self.map_origin_y = msg.info.origin.position.y

    def get_cell_cost(self, x: float, y: float) -> int:
        if self.map_data is None: return 100
        raw_col = int((x - self.map_origin_x) / self.map_res)
        raw_row = int((y - self.map_origin_y) / self.map_res)

        MARGIN = 12
        if raw_col < MARGIN or raw_col >= (self.map_width - MARGIN) or \
           raw_row < MARGIN or raw_row >= (self.map_height - MARGIN):
            return 100

        col = max(0, min(raw_col, self.map_width - 1))
        row = max(0, min(raw_row, self.map_height - 1))
        cost = self.map_data[row * self.map_width + col]
        return 100 if cost == -1 else cost

    def is_path_clear(self, start_x: float, start_y: float, end_x: float, end_y: float, max_cost: int = 45) -> bool:
        if self.map_data is None: return False
        dist = math.hypot(end_x - start_x, end_y - start_y)

        steps = max(2, int(dist / 0.08))
        t = np.linspace(0.0, 1.0, steps)
        px = start_x + t * (end_x - start_x)
        py = start_y + t * (end_y - start_y)

        raw_cols = ((px - self.map_origin_x) / self.map_res).astype(np.int32)
        raw_rows = ((py - self.map_origin_y) / self.map_res).astype(np.int32)

        MARGIN = 12
        if np.any(raw_cols < MARGIN) or np.any(raw_cols >= (self.map_width - MARGIN)) or \
           np.any(raw_rows < MARGIN) or np.any(raw_rows >= (self.map_height - MARGIN)):
            return False

        cols = np.clip(raw_cols, 0, self.map_width - 1)
        rows = np.clip(raw_rows, 0, self.map_height - 1)
        costs = self.map_data[rows * self.map_width + cols]

        if np.any(costs[1:] > max_cost) or np.any(costs == -1): return False
        return True

    def check_path_feasibility(self, start_x, start_y, end_x, end_y):
        # 1. Straight path check
        if self.is_path_clear(start_x, start_y, end_x, end_y, max_cost=self.MAX_SAFE_COST):
            return True, 'STRAIGHT'

        # 2. Dogleg circumvention around pillars
        direct_angle = math.atan2(end_y - start_y, end_x - start_x)
        total_dist = math.hypot(end_x - start_x, end_y - start_y)
        mid_dist = total_dist * 0.5

        for offset in [math.radians(45), math.radians(-45)]:
            mid_x = start_x + (mid_dist * math.cos(direct_angle + offset))
            mid_y = start_y + (mid_dist * math.sin(direct_angle + offset))

            if self.get_cell_cost(mid_x, mid_y) < 30:
                if self.is_path_clear(start_x, start_y, mid_x, mid_y, max_cost=self.MAX_SAFE_COST) and \
                   self.is_path_clear(mid_x, mid_y, end_x, end_y, max_cost=self.MAX_SAFE_COST):
                    return True, 'DOGLEG'

        return False, 'NONE'

    def get_safe_escape_points(self, robot_x, robot_y):
        safe_points = []
        for r in self.sample_radii:
            for cos_val, sin_val, rad in self.angle_steps:
                cand_x = robot_x + (r * cos_val)
                cand_y = robot_y + (r * sin_val)

                if self.get_cell_cost(cand_x, cand_y) > self.MAX_SAFE_COST:
                    continue

                is_clear, path_type = self.check_path_feasibility(robot_x, robot_y, cand_x, cand_y)
                if is_clear:
                    safe_points.append((cand_x, cand_y, r, self.get_cell_cost(cand_x, cand_y), path_type, rad))
        return safe_points

    def get_best_escape_target(self, safe_points, our_x, our_y, our_yaw, enemy_x, enemy_y):
        best_point, highest_score = None, -float('inf')
        current_cost = self.get_cell_cost(our_x, our_y)
        dist_to_enemy = math.hypot(our_x - enemy_x, our_y - enemy_y)

        # Dynamic weights: aggressive escape urgency when enemy is close
        w_dist = 18.0 if dist_to_enemy < 1.2 else 12.0
        w_speed = 4.0
        w_turn = 2.0
        w_cost = 0.35

        away_dx = our_x - enemy_x
        away_dy = our_y - enemy_y
        away_angle = math.atan2(away_dy, away_dx)

        if self.last_away_angle is not None:
            angle_shift = abs(math.atan2(math.sin(away_angle - self.last_away_angle), math.cos(away_angle - self.last_away_angle)))
            if angle_shift > math.radians(45.0):
                self.slide_direction_bias = None
        self.last_away_angle = away_angle

        # Multi-point check along direct away line to detect perimeter wall traps
        direct_away_blocked = False
        for step in [0.4, 0.8, 1.2]:
            chk_x = our_x + step * math.cos(away_angle)
            chk_y = our_y + step * math.sin(away_angle)
            if self.get_cell_cost(chk_x, chk_y) > 35:
                direct_away_blocked = True
                break

        tangent_left = away_angle + (math.pi / 2.0)
        tangent_right = away_angle - (math.pi / 2.0)

        for cand_x, cand_y, r, cost, path_type, cand_angle in safe_points:
            cand_enemy_dist = math.hypot(cand_x - enemy_x, cand_y - enemy_y)
            score = (cand_enemy_dist * w_dist) + (r * w_speed)

            # 1. TANGENTIAL WALL SLIDING
            if direct_away_blocked:
                diff_left = abs(math.atan2(math.sin(cand_angle - tangent_left), math.cos(cand_angle - tangent_left)))
                diff_right = abs(math.atan2(math.sin(cand_angle - tangent_right), math.cos(cand_angle - tangent_right)))
                slide_score = (math.pi - min(diff_left, diff_right)) * 14.0
                score += slide_score

                if self.slide_direction_bias == 'LEFT' and diff_left < diff_right:
                    score += 15.0
                elif self.slide_direction_bias == 'RIGHT' and diff_right < diff_left:
                    score += 15.0
            else:
                angle_diff_away = abs(math.atan2(math.sin(cand_angle - away_angle), math.cos(cand_angle - away_angle)))
                score += (math.pi - angle_diff_away) * 10.0

            # 2. DEEP SPRINT BONUS (r >= 1.4m gets bonus for maintaining momentum)
            if r >= 1.4:
                score += 12.0

            # 3. LIGHT LINEAR COST PENALTY (never chokes on narrow corridors)
            score -= cost * w_cost

            turn_penalty = abs(math.atan2(math.sin(cand_angle - our_yaw), math.cos(cand_angle - our_yaw)))
            score -= turn_penalty * w_turn

            if path_type == 'STRAIGHT':
                score += 8.0

            # Active target hysteresis to stop DWB trajectory oscillation
            if self.active_target_x is not None and self.active_target_y is not None:
                if math.hypot(cand_x - self.active_target_x, cand_y - self.active_target_y) < 0.5:
                    score += 15.0

            if score > highest_score:
                highest_score = score
                best_point = (cand_x, cand_y, cand_angle)

        if best_point is not None:
            self.active_target_x, self.active_target_y = best_point[0], best_point[1]
            cand_angle = best_point[2]
            diff_left = abs(math.atan2(math.sin(cand_angle - tangent_left), math.cos(cand_angle - tangent_left)))
            diff_right = abs(math.atan2(math.sin(cand_angle - tangent_right), math.cos(cand_angle - tangent_right)))
            self.slide_direction_bias = 'LEFT' if diff_left < diff_right else 'RIGHT'

        return best_point

    def execute_emergency_escape(self, global_x: float, global_y: float, enemy_x: float, enemy_y: float):
        for dist in [0.6, 1.0, 1.4]:
            for cos_val, sin_val, rad in self.angle_steps:
                cand_x = global_x + (dist * cos_val)
                cand_y = global_y + (dist * sin_val)
                if self.get_cell_cost(cand_x, cand_y) <= self.MAX_SAFE_COST:
                    if self.is_path_clear(global_x, global_y, cand_x, cand_y, max_cost=self.MAX_SAFE_COST):
                        return cand_x, cand_y, rad

        dx = global_x - enemy_x
        dy = global_y - enemy_y
        dist = math.hypot(dx, dy)
        if dist == 0: dist = 0.01

        for step in [0.5, 0.3, 0.2]:
            test_x = global_x + (dx / dist) * step
            test_y = global_y + (dy / dist) * step
            if self.get_cell_cost(test_x, test_y) < self.MAX_SAFE_COST:
                return test_x, test_y, math.atan2(dy, dx)

        return global_x, global_y, math.atan2(dy, dx)

    # TUNED SLEWING: Generous 75 deg/tick clamp lets the bot pivot into an escape route in 1 tick
    # without generating full 180 deg hairpin trajectory loops.
    def apply_angular_goal_slewing(self, robot_x: float, robot_y: float, target_x: float, target_y: float, dist_to_enemy: float, our_yaw: float):
        desired_angle = math.atan2(target_y - robot_y, target_x - robot_x)

        if self.last_goal_angle is None:
            self.last_goal_angle = our_yaw

        angle_diff = math.atan2(
            math.sin(desired_angle - self.last_goal_angle),
            math.cos(desired_angle - self.last_goal_angle)
        )

        # Under 1.2m: instantaneous break-away snap. Over 1.2m: crisp 75 deg/tick pivot
        max_step = math.radians(180.0) if dist_to_enemy < 1.2 else math.radians(75.0)

        clamped_diff = max(-max_step, min(max_step, angle_diff))
        slewed_angle = self.last_goal_angle + clamped_diff

        radius = math.hypot(target_x - robot_x, target_y - robot_y)
        slewed_x = robot_x + radius * math.cos(slewed_angle)
        slewed_y = robot_y + radius * math.sin(slewed_angle)

        if self.get_cell_cost(slewed_x, slewed_y) > self.MAX_SAFE_COST or \
           not self.is_path_clear(robot_x, robot_y, slewed_x, slewed_y, max_cost=self.MAX_SAFE_COST):
            self.last_goal_angle = desired_angle
            return target_x, target_y, desired_angle

        self.last_goal_angle = slewed_angle
        return slewed_x, slewed_y, slewed_angle

    def control_loop(self):
        if self.map_data is None:
            return

        try:
            tf_global = self.tf_buffer.lookup_transform(
                self.map_frame, self.robot_frame, rclpy.time.Time(), 
                timeout=rclpy.duration.Duration(seconds=0.02)
            )
            tf_enemy_global = self.tf_buffer.lookup_transform(
                self.map_frame, self.enemy_frame, rclpy.time.Time(), 
                timeout=rclpy.duration.Duration(seconds=0.02)
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

        target_x, target_y, target_yaw = self.apply_angular_goal_slewing(
            global_x, global_y, target_x, target_y, dist_to_enemy, our_yaw
        )

        self.last_target_x = target_x
        self.last_target_y = target_y

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.target_frame
        t.transform.translation.x = target_x
        t.transform.translation.y = target_y
        t.transform.translation.z = 0.0
        t.transform.rotation.z = math.sin(target_yaw / 2.0)
        t.transform.rotation.w = math.cos(target_yaw / 2.0)
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(EvasionCalculator())
    rclpy.shutdown()

if __name__ == '__main__':
    main()