#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
import tf2_ros
from geometry_msgs.msg import TransformStamped

class ChaserCalculator(Node):
    def __init__(self):
        super().__init__('chaser_calculator')

        self.map_frame = 'map'
        self.pursuer_frame = 'our_bot/base_link'
        self.carrot_frame = 'enemy_bot/target_point'
        self.broadcast_frame = 'chase_nav_target'

        # TUNING POINT: Intermediate goal lead distance horizon bounds.
        # Dynamic scaling: 1.2m for fast straight pursuit, down to 0.5m for tight cornering around pillars.
        self.max_goal_dist = 1.2
        self.min_goal_dist = 0.5
        
        # TUNING POINT: Terminal capture distance threshold (meters) to enforce enemy rotation alignment.
        # Keeps the robot driving 100% straight down corridors without creating parabolic curve arcs.
        self.capture_orientation_dist = 0.35

        self.current_state = 'CHASING'
        self.costmap = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.create_subscription(String, 'our_bot/robot_state', self.state_cb, 10)
        self.create_subscription(OccupancyGrid, '/our_bot/global_costmap/costmap', self.costmap_cb, 10)

        # TUNING POINT: 10 Hz (0.1s) is fine for smooth goal broadcasting.
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Chaser Calculator initialized. Ready for future attack logic.")

    def state_cb(self, msg):
        self.current_state = msg.data

    def costmap_cb(self, msg: OccupancyGrid):
        self.costmap = msg

    def is_pose_safe(self, x: float, y: float, max_cost: int = 50) -> bool:
        if self.costmap is None: return True
        info = self.costmap.info
        gx = int((x - info.origin.position.x) / info.resolution)
        gy = int((y - info.origin.position.y) / info.resolution)

        # MAP BOUNDARY GUARD: Enforce a 15-cell margin (0.75m) from map edges
        MARGIN = 15
        if gx < MARGIN or gx >= (info.width - MARGIN) or \
           gy < MARGIN or gy >= (info.height - MARGIN):
            return False

        cost = self.costmap.data[gy * info.width + gx]
        return 0 <= cost < max_cost

    def is_line_safe(self, start_x: float, start_y: float, target_x: float, target_y: float) -> bool:
        if self.costmap is None: return True
        dist = math.hypot(target_x - start_x, target_y - start_y)
        steps = max(2, int(dist / 0.05))
        for i in range(steps + 1):
            t = i / steps
            px = start_x + t * (target_x - start_x)
            py = start_y + t * (target_y - start_y)
            if not self.is_pose_safe(px, py, max_cost=50):
                return False
        return True

    def project_to_free_space(self, start_x: float, start_y: float, target_x: float, target_y: float):
        # 1. Direct path check
        if self.is_line_safe(start_x, start_y, target_x, target_y):
            return target_x, target_y

        # 2. Obstacle circumvention: If direct line crosses inflation/obstacle (e.g. pillar or enemy body),
        # try arc offsets around midpoint so we don't collapse directly backwards onto our bumper
        direct_dist = math.hypot(target_x - start_x, target_y - start_y)
        if direct_dist > 0.1:
            direct_angle = math.atan2(target_y - start_y, target_x - start_x)
            mid_dist = direct_dist * 0.5
            for offset in [math.radians(45), math.radians(-45), math.radians(60), math.radians(-60)]:
                cand_x = start_x + (mid_dist * math.cos(direct_angle + offset))
                cand_y = start_y + (mid_dist * math.sin(direct_angle + offset))
                if self.is_pose_safe(cand_x, cand_y, max_cost=20):
                    if self.is_line_safe(start_x, start_y, cand_x, cand_y) and self.is_pose_safe(cand_x, cand_y):
                        return cand_x, cand_y

        # 3. Fallback: Linear reduction towards start position
        if direct_dist < 0.05: return start_x, start_y
        for i in range(1, 15):
            alpha = 1.0 - (i / 15.0)
            px = start_x + alpha * (target_x - start_x)
            py = start_y + alpha * (target_y - start_y)
            if self.is_pose_safe(px, py, max_cost=50): 
                return px, py

        return start_x, start_y

    def control_loop(self):
        if self.current_state != 'CHASING':
            return

        try:
            t_our = self.tf_buffer.lookup_transform(self.map_frame, self.pursuer_frame, rclpy.time.Time())
            t_carrot = self.tf_buffer.lookup_transform(self.map_frame, self.carrot_frame, rclpy.time.Time())
        except tf2_ros.TransformException:
            return

        p_x, p_y = t_our.transform.translation.x, t_our.transform.translation.y
        c_x, c_y = t_carrot.transform.translation.x, t_carrot.transform.translation.y
        c_rot = t_carrot.transform.rotation

        dx = c_x - p_x
        dy = c_y - p_y
        dist = math.hypot(dx, dy)

        # DYNAMIC HORIZON: Scale lead distance dynamically based on distance to target
        if dist > 2.0:
            short_goal_dist = self.max_goal_dist  # 1.2m for long straight pursuit
        elif dist > 1.0:
            short_goal_dist = 0.8                  # 0.8m standard
        else:
            short_goal_dist = self.min_goal_dist  # 0.5m for tight cornering

        if dist > short_goal_dist:
            ratio = short_goal_dist / dist
            target_x = p_x + dx * ratio
            target_y = p_y + dy * ratio
        else:
            target_x = c_x
            target_y = c_y

        # Heading along vector of travel to force straight paths down hallways
        move_yaw = math.atan2(dy, dx)

        # Enforce enemy rotation ONLY within terminal capture distance (<= 0.35m)
        if dist <= self.capture_orientation_dist:
            rot_z = c_rot.z
            rot_w = c_rot.w
        else:
            rot_z = math.sin(move_yaw / 2.0)
            rot_w = math.cos(move_yaw / 2.0)

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