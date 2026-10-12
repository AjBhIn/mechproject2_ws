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
        self.state_sub = self.create_subscription(String, 'our_bot/robot_state', self.state_cb, 10)
        self.costmap_sub = self.create_subscription(OccupancyGrid, '/our_bot/global_costmap/costmap', self.costmap_callback, 10)

        # TACTICAL HORIZON: Short, highly-reachable immediate goals (0.5m - 1.2m)
        # Keeps goals well within DWB local window so the bot moves instantly without stalling.
        self.sample_radii = [0.5, 0.8, 1.2]
        
        # 15-degree steps (24 rays total) = ultra-fast computation (< 2ms)
        self.angle_steps = [(math.cos(math.radians(deg)), math.sin(math.radians(deg))) for deg in range(0, 360, 15)]

        self.map_data = None
        self.map_res = 0.05
        self.map_width = 0
        self.map_height = 0
        self.map_origin_x = 0.0 
        self.map_origin_y = 0.0
        
        # OccupancyGrid threshold (0-100 scale): <= 40 is safe navigable space
        self.MAX_SAFE_COST = 40

        self.last_target_x, self.last_target_y = None, None
        self.MIN_UPDATE_DIST = 0.03

        # Active target lock to prevent goal flickering during small movements
        self.active_target_x = None
        self.active_target_y = None

        self.timer = self.create_timer(0.1, self.control_loop)

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

        MARGIN = 10
        if raw_col < MARGIN or raw_col >= (self.map_width - MARGIN) or \
           raw_row < MARGIN or raw_row >= (self.map_height - MARGIN):
            return 100

        col = max(0, min(raw_col, self.map_width - 1))
        row = max(0, min(raw_row, self.map_height - 1))

        cost = self.map_data[row * self.map_width + col]
        return 100 if cost == -1 else cost

    # Direct line-of-sight raycast: fast, simple straight-line check
    def is_straight_path_clear(self, start_x: float, start_y: float, end_x: float, end_y: float) -> bool:
        if self.map_data is None: return False
        dist = math.hypot(end_x - start_x, end_y - start_y)

        steps = max(2, int(dist / 0.05))
        t = np.linspace(0.0, 1.0, steps)
        px = start_x + t * (end_x - start_x)
        py = start_y + t * (end_y - start_y)

        raw_cols = ((px - self.map_origin_x) / self.map_res).astype(np.int32)
        raw_rows = ((py - self.map_origin_y) / self.map_res).astype(np.int32)
        if np.any(raw_cols < 0) or np.any(raw_cols >= self.map_width) or \
           np.any(raw_rows < 0) or np.any(raw_rows >= self.map_height):
            return False

        indices = raw_rows * self.map_width + raw_cols
        costs = self.map_data[indices]

        return not (np.any(costs > self.MAX_SAFE_COST) or np.any(costs == -1))

    def get_tactical_escape_target(self, our_x, our_y, our_yaw, enemy_x, enemy_y):
        best_point = None
        highest_score = -float('inf')

        # Vector pointing directly away from enemy
        away_dx = our_x - enemy_x
        away_dy = our_y - enemy_y
        away_angle = math.atan2(away_dy, away_dx)

        for r in self.sample_radii:
            for cos_val, sin_val in self.angle_steps:
                cand_x = our_x + (r * cos_val)
                cand_y = our_y + (r * sin_val)

                # Only evaluate points with a CLEAR straight-line path from our bot
                if not self.is_straight_path_clear(our_x, our_y, cand_x, cand_y):
                    continue

                cost = self.get_cell_cost(cand_x, cand_y)
                cand_angle = math.atan2(cand_y - our_y, cand_x - our_x)

                # 1. Primary Objective: Align vector away from enemy
                angle_diff_from_away = abs(math.atan2(math.sin(cand_angle - away_angle), math.cos(cand_angle - away_angle)))
                away_alignment_score = (math.pi - angle_diff_from_away) * 10.0

                # 2. Distance from Enemy score
                dist_to_enemy = math.hypot(cand_x - enemy_x, cand_y - enemy_y)
                dist_score = dist_to_enemy * 15.0

                # 3. Turn Penalty (prefer directions easy for current robot orientation)
                turn_angle = abs(math.atan2(math.sin(cand_angle - our_yaw), math.cos(cand_angle - our_yaw)))
                turn_penalty = turn_angle * 3.0

                score = away_alignment_score + dist_score - turn_penalty - (cost * 0.3)

                # Active target hysteresis (+10.0) to maintain commitment during motion
                if self.active_target_x is not None and self.active_target_y is not None:
                    if math.hypot(cand_x - self.active_target_x, cand_y - self.active_target_y) < 0.3:
                        score += 10.0

                if score > highest_score:
                    highest_score = score
                    best_point = (cand_x, cand_y, cand_angle)

        if best_point is not None:
            self.active_target_x, self.active_target_y = best_point[0], best_point[1]

        return best_point

    # Fallback if surrounded: find shortest free ray to step out of immediate collision
    def get_emergency_step_aside(self, our_x, our_y, enemy_x, enemy_y):
        for dist in [0.4, 0.6]:
            for cos_val, sin_val in self.angle_steps:
                cand_x = our_x + (dist * cos_val)
                cand_y = our_y + (dist * sin_val)
                if self.is_straight_path_clear(our_x, our_y, cand_x, cand_y):
                    return cand_x, cand_y, math.atan2(sin_val, cos_val)

        # Absolute fallback away from enemy
        dx = our_x - enemy_x
        dy = our_y - enemy_y
        d = max(0.01, math.hypot(dx, dy))
        return our_x + (dx / d) * 0.4, our_y + (dy / d) * 0.4, math.atan2(dy, dx)

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

        best_target = self.get_tactical_escape_target(global_x, global_y, our_yaw, enemy_global_x, enemy_global_y)

        if best_target is None:
            target_x, target_y, target_yaw = self.get_emergency_step_aside(global_x, global_y, enemy_global_x, enemy_global_y)
        else:
            target_x, target_y, target_yaw = best_target

        # Light exponential filter to smooth target position
        if self.last_target_x is not None and self.last_target_y is not None:
            target_x = 0.5 * target_x + 0.5 * self.last_target_x
            target_y = 0.5 * target_y + 0.5 * self.last_target_y

        if self.last_target_x is not None:
            if math.hypot(target_x - self.last_target_x, target_y - self.last_target_y) < self.MIN_UPDATE_DIST:
                return

        self.last_target_x, self.last_target_y = target_x, target_y

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.target_frame
        t.transform.translation.x = target_x
        t.transform.translation.y = target_y
        t.transform.rotation.z = math.sin(target_yaw / 2.0)
        t.transform.rotation.w = math.cos(target_yaw / 2.0)

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(EvasionCalculator())
    rclpy.shutdown()

if __name__ == '__main__':
    main()