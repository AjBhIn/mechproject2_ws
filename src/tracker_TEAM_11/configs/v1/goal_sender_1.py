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
try:
    from tracker_msgs.msg import TrackerReport
    HAS_TRACKER_MSGS = True
except ImportError:
    HAS_TRACKER_MSGS = False
    print("\n[CRITICAL WARNING]: 'tracker_msgs' not found! Reports will NOT be published.\n", file=sys.stderr)

# --- GLOBAL TUNING VARIABLES ---
LOOP_RATE_SEC = 0.05
ACTION_SERVER_TIMEOUT = 1.0       # Relaxed from 0.01s for physical WiFi

# Capture (Kill) Thresholds
CAPTURE_DISTANCE_M = 0.50         # Manual capture trigger distance (bypasses Nav2 success)
CAPTURE_COOLDOWN_SEC = 5.0        # Prevents spamming RViz with hundreds of photos of the same kill

# Goal Throttling (Don't spam Nav2 if the goal hasn't moved much)
MIN_GOAL_MOVE_DIST = 0.35         
MIN_GOAL_YAW_CHANGE = math.radians(15.0)
MIN_GOAL_TIME_ELAPSED = 0.60

class GoalSender(Node):
    """
    GoalSender tracks the dynamic targets, dispatches throttled PoseStamped 
    goals to Nav2, and manually triggers capture reports when close enough.
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

        try:
            self.my_id_val = int(self.robot_ns[-2:]) if len(self.robot_ns) > 2 else 1
            self.opp_id_val = int(self.enemy_ns[-2:]) if len(self.enemy_ns) > 2 else 2
        except ValueError:
            self.my_id_val, self.opp_id_val = 1, 2

        # --- Internal State ---
        self.last_goal_time = 0.0
        self.last_capture_time = 0.0
        self.state = 'CHASING'
        self.last_sent_pose = None
        self.active_goal_handle = None
        self.is_goal_pending = False
        self.force_update = False 

        # --- Camera Setup ---
        self.tracker_image = None
        camera_topic = f'/{self.robot_ns}/oakd/rgb/preview/image_raw/compressed' if self.robot_ns else '/oakd/rgb/preview/image_raw/compressed'
        self.image_sub = self.create_subscription(CompressedImage, camera_topic, self.image_callback, 10)

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
        
        self.timer = self.create_timer(LOOP_RATE_SEC, self.control_loop)
        self.get_logger().info("Goal Sender initialized. Waiting for Nav2 Action Server...")

    def image_callback(self, msg):
        self.tracker_image = msg

    def state_cb(self, msg):
        new_state = msg.data
        if self.state != new_state:
            self.get_logger().info(f"State transitioned to {new_state}. Forcing goal update.")
            self.state = new_state
            self.force_update = True 

    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def trigger_capture_report(self, rel_dist, rel_orient):
        """Builds and publishes the TrackerReport exactly as expected by RViz."""
        if not HAS_TRACKER_MSGS or self.tracker_image is None: return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        if (now_sec - self.last_capture_time) < CAPTURE_COOLDOWN_SEC:
            return  # Cooldown active, don't spam

        try:
            report = TrackerReport()
            report.header.stamp = self.get_clock().now().to_msg()
            report.header.frame_id = str(self.map_frame)
            
            # Copy image fields straight from the CompressedImage
            report.image.header = self.tracker_image.header
            report.image.format = self.tracker_image.format
            report.image.data = self.tracker_image.data
            
            report.my_id = self.my_id_val & 0xFF
            report.opp_id = self.opp_id_val & 0xFF
            report.rel_dist = float(rel_dist)
            report.rel_orient = float(rel_orient)

            self.report_pub.publish(report)
            self.report_img_pub.publish(self.tracker_image)
            self.last_capture_time = now_sec
            self.get_logger().info(f">>> CAPTURE REPORT SENT! Distance: {rel_dist:.2f}m <<<")
            
            # Cancel current navigation goal to stop oscillation
            if self.active_goal_handle is not None:
                self.active_goal_handle.cancel_goal_async()
                
        except Exception as e:
            self.get_logger().error(f"[ERROR] capture report NOT sent - {type(e).__name__}: {e}")

    def goal_response_callback(self, future):
        self.is_goal_pending = False
        try:
            goal_handle = future.result()
            if goal_handle.accepted:
                self.active_goal_handle = goal_handle
            else:
                self.active_goal_handle = None
        except Exception:
            self.active_goal_handle = None

    def control_loop(self):
        if not self.nav_client.wait_for_server(timeout_sec=ACTION_SERVER_TIMEOUT): 
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9

        # --- Capture Evaluation Block ---
        if self.state == 'CHASING':
            try:
                rel_tf = self.tf_buffer.lookup_transform(
                    self.robot_frame, self.enemy_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)
                )
                rel_dist = math.hypot(rel_tf.transform.translation.x, rel_tf.transform.translation.y)
                if rel_dist <= CAPTURE_DISTANCE_M:
                    rel_orient = self.get_yaw_from_quaternion(rel_tf.transform.rotation)
                    self.trigger_capture_report(rel_dist, rel_orient)
            except tf2_ros.TransformException:
                pass

        # --- Goal Dispatch Block ---
        if self.force_update:
            self.is_goal_pending = False

        if self.is_goal_pending: 
            if (now_sec - self.last_goal_time) > 1.0: # Goal timed out
                self.is_goal_pending = False
            else:
                return

        target_frame = self.escape_frame if self.state == 'EVADING' else self.chase_frame
        try:
            target_tf = self.tf_buffer.lookup_transform(
                self.map_frame, target_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except tf2_ros.TransformException:
            return

        t_x = target_tf.transform.translation.x
        t_y = target_tf.transform.translation.y
        t_rot = target_tf.transform.rotation

        # Throttle goals if we haven't moved much to keep planner from stuttering
        if not self.force_update and self.last_sent_pose is not None:
            dist_moved = math.hypot(t_x - self.last_sent_pose.pose.position.x, t_y - self.last_sent_pose.pose.position.y)
            time_elapsed = now_sec - self.last_goal_time

            yaw_current = self.get_yaw_from_quaternion(t_rot)
            yaw_last = self.get_yaw_from_quaternion(self.last_sent_pose.pose.orientation)
            yaw_change = abs(math.atan2(math.sin(yaw_current - yaw_last), math.cos(yaw_current - yaw_last)))

            if dist_moved < MIN_GOAL_MOVE_DIST and yaw_change < MIN_GOAL_YAW_CHANGE and time_elapsed < MIN_GOAL_TIME_ELAPSED:
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