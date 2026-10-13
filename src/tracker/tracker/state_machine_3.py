#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import tf2_ros

class StateMachine(Node):
    def __init__(self):
        super().__init__('state_machine')

        self.state = 'CHASING'

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Created our own publisher for robot state to avoid circular dependency with goal_sender
        self.state_pub = self.create_publisher(String, 'our_bot/robot_state', 10)

        # ENHANCEMENT: Hold time ONLY applies when exiting EVADING so evasion bursts complete,
        # but CHASING -> EVADING transitions are INSTANT.
        self.min_evasion_duration = 0.6  
        self.last_evasion_start_time = 0.0

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("State Machine Initialized (High-Sensitivity Threat Curve).")

    def get_pursuer_in_enemy_frame(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                'enemy_bot/base_link', 'our_bot/base_link', 
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.05)
            )
            return trans.transform.translation.x, trans.transform.translation.y
        except tf2_ros.TransformException:
            return None, None

    # Checking if our bot is in the danger zone of the enemy bot using continuous angular interpolation
    def evaluate_threat_state(self, rel_x: float, rel_y: float):
        distance = math.hypot(rel_x, rel_y)
        relative_angle_deg = abs(math.degrees(math.atan2(rel_y, rel_x)))

        # SENSITIVE THREAT ENVELOPE:
        # Front (0 deg)   -> 3.0m threshold
        # Flank (90 deg)  -> 2.0m threshold
        # Rear (180 deg)  -> 0.0m threshold (safe behind)
        if relative_angle_deg <= 90.0:
            ratio = relative_angle_deg / 90.0
            current_threshold = 3.0 - (1.0 * ratio)
        else:
            ratio = (relative_angle_deg - 90.0) / 90.0
            current_threshold = 2.0 - (2.0 * ratio)
            
        if distance > current_threshold:
            return False, distance, current_threshold
            
        return True, distance, current_threshold

    def control_loop(self):
        # Get the relative position of our bot in the enemy bot's frame
        rel_x, rel_y = self.get_pursuer_in_enemy_frame()
        if rel_x is None: 
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        time_in_evasion = now_sec - self.last_evasion_start_time

        in_danger, dist, threshold = self.evaluate_threat_state(rel_x, rel_y)

        if self.state == 'CHASING':
            # INSTANT SENSITIVITY: Switch to EVADING immediately on threat detection (no lockout delay)
            if in_danger:
                self.get_logger().warn(f'DANGER ZONE! Dist: {dist:.2f}m <= {threshold:.2f}m. Switching to EVADING.')
                self.state = 'EVADING'
                self.last_evasion_start_time = now_sec

        elif self.state == 'EVADING':
            # Require robot to clear threat + 0.5m hysteresis AND hold evasion for at least 0.6s
            if not in_danger and dist > (threshold + 0.5) and time_in_evasion >= self.min_evasion_duration:
                self.get_logger().info(f'Clear of threat (Dist: {dist:.2f}m). Resuming CHASING.')
                self.state = 'CHASING'

        msg = String()
        msg.data = self.state
        self.state_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(StateMachine())
    rclpy.shutdown()

if __name__ == '__main__':
    main()