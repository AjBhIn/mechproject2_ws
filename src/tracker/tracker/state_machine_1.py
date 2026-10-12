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

        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("State Machine Initialized (Hardened TF Sync).")

    def get_pursuer_in_enemy_frame(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                'enemy_bot/base_link', 'our_bot/base_link', 
                rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.05)
            )
            return trans.transform.translation.x, trans.transform.translation.y
        except tf2_ros.TransformException:
            return None, None

    # Checking if our bot is in the danger zone of the enemy bot
    def evaluate_threat_state(self, rel_x: float, rel_y: float):
        distance = math.hypot(rel_x, rel_y)
        relative_angle_deg = abs(math.degrees(math.atan2(rel_y, rel_x)))

        # TUNING POINT: Threat angle cones. 
        # Narrow these if your bot is too skittish on the sides.
        in_front_cone = relative_angle_deg <= 67.5       
        in_back_cone = relative_angle_deg >= 135.0       
        
        # TUNING POINT: Threat Distances.
        if in_front_cone:
            current_threshold = 2.8   # Danger distance when facing the enemy's front
        elif in_back_cone:
            current_threshold = 0.0   # Danger distance when behind the enemy (0 = always safe)
        else:
            current_threshold = 2.0   # Danger distance on the flanks
            
        if distance > current_threshold:
            return False, distance, current_threshold
            
        return True, distance, current_threshold

    def control_loop(self):
        # Get the relative position of our bot in the enemy bot's frame
        rel_x, rel_y = self.get_pursuer_in_enemy_frame()
        if rel_x is None: 
            return

        in_danger, dist, threshold = self.evaluate_threat_state(rel_x, rel_y)

        if self.state == 'CHASING':
            if in_danger:
                self.get_logger().warn(f'DANGER ZONE! Dist: {dist:.2f}m <= {threshold}m. Switching to EVADING.')
                self.state = 'EVADING'

        elif self.state == 'EVADING':
            # TUNING POINT: The + 0.3 is the hysteresis buffer. 
            # If your bot rapidly flips between chasing/evading on the edge of the threat zone, increase this to 0.5.
            if not in_danger and dist > (threshold + 0.3):
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