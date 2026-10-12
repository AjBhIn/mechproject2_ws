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

        # TUNING POINT: How far ahead along the line-of-sight the intermediate goal is placed.
        # Reduce to 0.5m for tighter turning, increase to 1.2m for faster, smoother straight lines.
        self.short_goal_dist = 0.8
        self.current_state = 'CHASING'
        self.costmap = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.create_subscription(String, 'our_bot/robot_state', self.state_cb, 10)
        self.create_subscription(OccupancyGrid, '/our_bot/global_costmap/costmap', self.costmap_cb, 10)

        # TUNING POINT: 10 Hz (0.1s) is usually fine, but if tracking fast bots, you can push this to 0.05 (20 Hz).
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Chaser Calculator initialized. Ready for future attack logic.")

    def state_cb(self, msg):
        self.current_state = msg.data

    def costmap_cb(self, msg: OccupancyGrid):
        self.costmap = msg

    def is_pose_safe(self, x: float, y: float) -> bool:
        if self.costmap is None: return True
        info = self.costmap.info
        gx = int((x - info.origin.position.x) / info.resolution)
        gy = int((y - info.origin.position.y) / info.resolution)
        if 0 <= gx < info.width and 0 <= gy < info.height:
            # TUNING POINT: 180 is the cost threshold for "safe". If the physical bot clips walls, lower this to 150 or 120.
            return self.costmap.data[gy * info.width + gx] < 180
        return False

    def project_to_free_space(self, start_x: float, start_y: float, target_x: float, target_y: float):
        if self.is_pose_safe(target_x, target_y): return target_x, target_y
        if math.hypot(target_x - start_x, target_y - start_y) < 0.05: return start_x, start_y
        for i in range(1, 10):
            alpha = 1.0 - (i / 10.0)
            px = start_x + alpha * (target_x - start_x)
            py = start_y + alpha * (target_y - start_y)
            if self.is_pose_safe(px, py): return px, py
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

        if dist > self.short_goal_dist:
            ratio = self.short_goal_dist / dist
            target_x = p_x + dx * ratio
            target_y = p_y + dy * ratio
        else:
            target_x = c_x
            target_y = c_y

        safe_x, safe_y = self.project_to_free_space(p_x, p_y, target_x, target_y)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.broadcast_frame
        t.transform.translation.x = safe_x
        t.transform.translation.y = safe_y
        t.transform.rotation = c_rot

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ChaserCalculator())
    rclpy.shutdown()

if __name__ == '__main__':
    main()