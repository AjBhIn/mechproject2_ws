#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from sensor_msgs.msg import CompressedImage
import tf2_ros
import sys

# --- ROBUST IMPORT ---
# Prevents node crash if tracker_msgs is missing/un-sourced
try:
    from tracker_msgs.msg import TrackerReport
    HAS_TRACKER_MSGS = True
except ImportError:
    HAS_TRACKER_MSGS = False
    print("\n[CRITICAL WARNING]: 'tracker_msgs' not found! The node will run and send goals, but TrackerReports will NOT be published.\n", file=sys.stderr)

class GoalSender(Node):
    """
    GoalSender tracks the dynamic targets and dispatches throttled PoseStamped 
    goals to the Nav2 Action Server. It publishes TrackerReports upon success.
    """
    def __init__(self):
        super().__init__('goal_sender')

        # --- Dynamic Namespace Setup ---
        raw_ns = self.get_namespace().strip('/')
        default_robot_ns = raw_ns if raw_ns else 'our_bot'

        self.declare_parameter('chaser', default_robot_ns)
        self.declare_parameter('enemy', 'enemy_bot')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('robot_frame', '')
        self.declare_parameter('action_name', '')
        self.declare_parameter('chase_frame', 'chase_nav_target')
        self.declare_parameter('escape_frame', 'dynamic_nav_target')

        self.robot_ns = self.get_parameter('chaser').value.strip('/')
        self.enemy_ns = self.get_parameter('enemy').value.strip('/')
        self.map_frame = self.get_parameter('map_frame').value
        
        robot_p = self.get_parameter('robot_frame').value
        self.robot_frame = robot_p if robot_p else (f'{self.robot_ns}/base_link' if self.robot_ns else 'base_link')
        self.enemy_frame = f'{self.enemy_ns}/base_link' if self.enemy_ns else 'base_link'

        action_p = self.get_parameter('action_name').value
        self.action_name = action_p if action_p else (f'/{self.robot_ns}/navigate_to_pose' if self.robot_ns else '/navigate_to_pose')

        self.chase_frame = self.get_parameter('chase_frame').value
        self.escape_frame = self.get_parameter('escape_frame').value

        # --- Internal State ---
        self.last_goal_time = 0.0
        self.state = 'CHASING'
        self.last_sent_pose = None
        self.active_goal_handle = None
        self.is_goal_pending = False
        self.force_update = False 

        # --- Camera Setup ---
        self.tracker_image = None
        camera_topic = f'/{self.robot_ns}/oakd/rgb/preview/image_raw/compressed' if self.robot_ns else '/oakd/rgb/preview/image_raw/compressed'
        self.image_sub = self.create_subscription(CompressedImage, camera_topic, self.image_cb, 10)

        # Only register custom publishers if the message package was successfully imported
        if HAS_TRACKER_MSGS:
            report_topic = f'/{self.robot_ns}/tracker_report' if self.robot_ns else '/tracker_report'
            self.report_pub = self.create_publisher(TrackerReport, report_topic, 10)

            report_img_topic = f'/{self.robot_ns}/tracker_report_img/compressed' if self.robot_ns else '/tracker_report_img/compressed'
            self.report_img_pub = self.create_publisher(CompressedImage, report_img_topic, 10)

        # --- TF2 & Action Client ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        state_topic = f'/{self.robot_ns}/robot_state' if self.robot_ns else '/robot_state'
        self.state_sub = self.create_subscription(String, state_topic, self.state_cb, 10)
        
        self.nav_client = ActionClient(self, NavigateToPose, self.action_name)
        
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info(f"Goal Sender initialized. Waiting for Nav2 Action Server...")

    def image_cb(self, msg):
        self.tracker_image = msg

    def state_cb(self, msg):
        new_state = msg.data
        if self.state != new_state:
            self.state = new_state
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
                result_future = goal_handle.get_result_async()
                result_future.add_done_callback(self.goal_result_callback)
            else:
                self.active_goal_handle = None
        except Exception as e:
            self.active_goal_handle = None

    def goal_result_callback(self, future):
        result = future.result()
        if result.status == GoalStatus.STATUS_SUCCEEDED:
            if not HAS_TRACKER_MSGS: return 
            if self.tracker_image is None: return

            try:
                rel_tf = self.tf_buffer.lookup_transform(
                    self.robot_frame, self.enemy_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)
                )
                rel_x = rel_tf.transform.translation.x
                rel_y = rel_tf.transform.translation.y
                rel_dist = math.hypot(rel_x, rel_y)
                rel_orient = math.degrees(self.get_yaw_from_quaternion(rel_tf.transform.rotation))
            except tf2_ros.TransformException:
                return

            try:
                my_id_val = int(self.robot_ns[-2:]) if len(self.robot_ns) > 2 else 1
                opp_id_val = int(self.enemy_ns[-2:]) if len(self.enemy_ns) > 2 else 2
            except ValueError:
                my_id_val, opp_id_val = 1, 2

            report = TrackerReport()
            report.image = self.tracker_image
            report.my_id = my_id_val
            report.opp_id = opp_id_val
            report.rel_dist = float(rel_dist)
            report.rel_orient = float(rel_orient)

            self.report_pub.publish(report)
            self.report_img_pub.publish(self.tracker_image)

    def control_loop(self):
        if not self.nav_client.wait_for_server(timeout_sec=0.01): return

        now_sec = self.get_clock().now().nanoseconds / 1e9

        if self.force_update:
            self.is_goal_pending = False

        if self.is_goal_pending: 
            if (now_sec - self.last_goal_time) > 0.5:
                self.is_goal_pending = False
            else:
                return

        target_tf = self.get_target_tf()
        if target_tf is None: return

        t_x = target_tf.transform.translation.x
        t_y = target_tf.transform.translation.y
        t_rot = target_tf.transform.rotation

        try:
            robot_tf = self.tf_buffer.lookup_transform(self.map_frame, self.robot_frame, rclpy.time.Time())
            dist_to_target = math.hypot(t_x - robot_tf.transform.translation.x, t_y - robot_tf.transform.translation.y)
        except tf2_ros.TransformException:
            dist_to_target = 2.0

        if dist_to_target < 1.2:
            dynamic_min_dist = 0.15   
            dynamic_min_time = 0.40   
        else:
            dynamic_min_dist = 0.35   
            dynamic_min_time = 0.60   

        if not self.force_update:
            if self.last_sent_pose is not None:
                dx = t_x - self.last_sent_pose.pose.position.x
                dy = t_y - self.last_sent_pose.pose.position.y
                dist_moved = math.hypot(dx, dy)
                time_elapsed = now_sec - self.last_goal_time

                yaw_current = self.get_yaw_from_quaternion(t_rot)
                yaw_last = self.get_yaw_from_quaternion(self.last_sent_pose.pose.orientation)
                yaw_change = abs(math.atan2(math.sin(yaw_current - yaw_last), math.cos(yaw_current - yaw_last)))

                if dist_moved < dynamic_min_dist and yaw_change < math.radians(15.0) and time_elapsed < dynamic_min_time:
                    return

        self.force_update = False

        goal = PoseStamped()
        goal.header.frame_id = self.map_frame
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = t_x
        goal.pose.position.y = t_y
        goal.pose.orientation = t_rot

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = goal

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