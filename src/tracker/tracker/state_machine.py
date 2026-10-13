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

        # ENHANCEMENT: Debounce timer to ensure robot commits to state bursts without chattering
        self.min_state_duration = 0.6  # Hold state for at least 0.6 seconds
        
        # NEW: Properly initialize time tracking variables
        # Using 0.0 can cause issues with ROS 2 clock initialization depending on the time source.
        self.last_state_change_time = None
        self.last_tf_success_time = None
        
        # NEW: Timeout duration (seconds) before assuming the enemy is lost
        self.tf_timeout_duration = 2.0 

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("State Machine Initialized (Smooth Angular Threat Curve).")

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

        # CONTINUOUS THREAT INTERPOLATION:
        # Front (0 deg)  -> 2.6m threshold
        # Flank (90 deg)  -> 1.5m threshold
        # Rear (180 deg)  -> 0.0m threshold (safe behind)
        if relative_angle_deg <= 90.0:
            ratio = relative_angle_deg / 90.0
            current_threshold = 2.6 - (1.1 * ratio)
        else:
            ratio = (relative_angle_deg - 90.0) / 90.0
            current_threshold = 1.5 - (1.5 * ratio)
            
        if distance > current_threshold:
            return False, distance, current_threshold
            
        return True, distance, current_threshold

    def control_loop(self):
        now_sec = self.get_clock().now().nanoseconds / 1e9
        
        # Initialize timestamps on first successful loop
        if self.last_state_change_time is None:
            self.last_state_change_time = now_sec
            self.last_tf_success_time = now_sec

        time_in_state = now_sec - self.last_state_change_time

        # Get the relative position of our bot in the enemy bot's frame
        rel_x, rel_y = self.get_pursuer_in_enemy_frame()
        
        # NEW: Handle Missing Enemy / Sensor Loss safely
        if rel_x is None: 
            time_since_last_tf = now_sec - self.last_tf_success_time
            if time_since_last_tf > self.tf_timeout_duration and self.state != 'CHASING':
                self.get_logger().warn(f'Enemy tracking lost for {time_since_last_tf:.1f}s. Reverting to CHASING (Search).')
                self.state = 'CHASING'
                self.last_state_change_time = now_sec
            
            # Publish current state and exit early
            msg = String()
            msg.data = self.state
            self.state_pub.publish(msg)
            return

        # Record successful TF lookup
        self.last_tf_success_time = now_sec

        in_danger, dist, threshold = self.evaluate_threat_state(rel_x, rel_y)

        if self.state == 'CHASING':
            # NEW: Asymmetrical Debouncing.
            # We REMOVED the `time_in_state >= self.min_state_duration` check here.
            # If the robot is in danger, it must evade INSTANTLY. Survival cannot wait for a debounce timer.
            if in_danger:
                self.get_logger().warn(f'DANGER ZONE! Dist: {dist:.2f}m <= {threshold:.2f}m. Switching to EVADING.')
                self.state = 'EVADING'
                self.last_state_change_time = now_sec

        elif self.state == 'EVADING':
            # TUNING POINT: Exit hysteresis buffer (+0.2m) and hold time check to prevent state chatter
            # NEW: Reduced hysteresis from +0.5m to +0.2m to prevent the "Hysteresis Trap",
            # allowing the robot to resume chasing without having to retreat exceptionally far if the enemy turns towards it.
            if not in_danger and dist > (threshold + 0.2) and time_in_state >= self.min_state_duration:
                self.get_logger().info(f'Clear of threat (Dist: {dist:.2f}m). Resuming CHASING.')
                self.state = 'CHASING'
                self.last_state_change_time = now_sec

        msg = String()
        msg.data = self.state
        self.state_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(StateMachine())
    rclpy.shutdown()

if __name__ == '__main__':
    main()