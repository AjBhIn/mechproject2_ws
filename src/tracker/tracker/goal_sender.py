#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
import tf2_ros

class GoalSender(Node):
    def __init__(self):
        super().__init__('goal_sender')

        self.map_frame = 'map'
        self.action_name = 'our_bot/navigate_to_pose'

        self.chase_frame = 'chase_nav_target'
        self.escape_frame = 'dynamic_nav_target'

        self.last_goal_time = 0.0
        self.state = 'CHASING'
        self.last_sent_pose = None
        self.active_goal_handle = None
        self.is_goal_pending = False
        
        # Thread-safe flag to force immediate update on state change without deadlocking
        self.force_update = False 

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.state_sub = self.create_subscription(String, 'our_bot/robot_state', self.state_cb, 10)
        self.nav_client = ActionClient(self, NavigateToPose, self.action_name)
        
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("Robust Goal Sender initialized with NATIVE preemption.")

    def state_cb(self, msg):
        new_state = msg.data
        if self.state != new_state:
            self.get_logger().info(f"State transition: {self.state} -> {new_state}. Utilizing Nav2 preemption.")
            self.state = new_state
            # Trigger immediate goal send on next loop instead of manually canceling
            self.force_update = True 

    def get_target_tf(self):
        target_frame = self.escape_frame if self.state == 'EVADING' else self.chase_frame
        try:
            trans = self.tf_buffer.lookup_transform(
                self.map_frame, target_frame, rclpy.time.Time(), 
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
            return trans
        except tf2_ros.TransformException:
            return None

    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def goal_response_callback(self, future):
        self.is_goal_pending = False
        try:
            goal_handle = future.result()
            if goal_handle.accepted:
                self.active_goal_handle = goal_handle
            else:
                self.active_goal_handle = None
        except Exception as e:
            self.get_logger().warn(f"Goal dispatch exception: {e}")
            self.active_goal_handle = None

    def control_loop(self):
        if not self.nav_client.wait_for_server(timeout_sec=0.01): 
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9

        # ENHANCEMENT 1: State Change Override
        # If a state change occurs, override pending status to force immediate goal dispatch
        if self.force_update:
            self.is_goal_pending = False

        # ENHANCEMENT 2: Deadlock Guard Timeout (0.5s timeout for pending action responses)
        if self.is_goal_pending: 
            if (now_sec - self.last_goal_time) > 0.5:
                self.get_logger().warn("Goal response timed out (>0.5s). Resetting pending flag.")
                self.is_goal_pending = False
            else:
                return

        target_tf = self.get_target_tf()
        if target_tf is None: 
            return

        t_x = target_tf.transform.translation.x
        t_y = target_tf.transform.translation.y
        t_rot = target_tf.transform.rotation

        # Get the robot's current position in the map frame to compute distance to target
        try:
            robot_tf = self.tf_buffer.lookup_transform(self.map_frame, 'our_bot/base_link', rclpy.time.Time())
            dist_to_target = math.hypot(t_x - robot_tf.transform.translation.x, t_y - robot_tf.transform.translation.y)
        except tf2_ros.TransformException:
            dist_to_target = 2.0

        # ENHANCEMENT 3: Optimized pacing thresholds (smooth execution without DWB starvation)
        if dist_to_target < 1.2:
            dynamic_min_dist = 0.15   
            dynamic_min_time = 0.30   
        else:
            dynamic_min_dist = 0.35   
            dynamic_min_time = 0.50   

        # Only apply pacing limits if we aren't being forced to update by a state change
        if not self.force_update:
            if self.last_sent_pose is not None:
                dx = t_x - self.last_sent_pose.pose.position.x
                dy = t_y - self.last_sent_pose.pose.position.y
                dist_moved = math.hypot(dx, dy)
                time_elapsed = now_sec - self.last_goal_time

                # ENHANCEMENT 4: Orientation Delta Check (trigger update if target rotates > 15 deg)
                yaw_current = self.get_yaw_from_quaternion(t_rot)
                yaw_last = self.get_yaw_from_quaternion(self.last_sent_pose.pose.orientation)
                yaw_change = abs(math.atan2(math.sin(yaw_current - yaw_last), math.cos(yaw_current - yaw_last)))

                if dist_moved < dynamic_min_dist and yaw_change < math.radians(15.0) and time_elapsed < dynamic_min_time:
                    return

        # Reset flag and commit to sending goal
        self.force_update = False

        goal = PoseStamped()
        goal.header.frame_id = self.map_frame
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = t_x
        goal.pose.position.y = t_y
        goal.pose.orientation = t_rot

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal

        # Saving the previous state
        self.is_goal_pending = True
        self.last_goal_time = now_sec
        self.last_sent_pose = goal

        send_future = self.nav_client.send_goal_async(goal_msg)
        send_future.add_done_callback(self.goal_response_callback)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(GoalSender())
    rclpy.shutdown()

if __name__ == '__main__':
    main()