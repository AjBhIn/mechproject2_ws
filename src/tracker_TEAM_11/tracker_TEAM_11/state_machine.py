#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import tf2_ros

class StateMachine(Node):
    """
    Velocity-Aware State Machine. Determines if the robot should be CHASING or EVADING 
    based on distance, relative angle, and the enemy's approach speed.
    """
    def __init__(self):
        super().__init__('state_machine')

        # --- Dynamic Namespace Setup ---
        raw_ns = self.get_namespace().strip('/')
        default_robot_ns = raw_ns if raw_ns else 'our_bot'

        self.declare_parameter('chaser', default_robot_ns)
        self.declare_parameter('enemy', 'enemy_bot')
        self.declare_parameter('pursuer_frame', '')
        self.declare_parameter('enemy_frame', '')

        self.robot_ns = self.get_parameter('chaser').value.strip('/')
        self.enemy_ns = self.get_parameter('enemy').value.strip('/')

        pursuer_p = self.get_parameter('pursuer_frame').value
        self.pursuer_frame = pursuer_p if pursuer_p else (f'{self.robot_ns}/base_link' if self.robot_ns else 'base_link')

        enemy_p = self.get_parameter('enemy_frame').value
        self.enemy_frame = enemy_p if enemy_p else (f'{self.enemy_ns}/base_link' if self.enemy_ns else 'base_link')

        self.state = 'CHASING'

        # --- TF2 & Subscriptions ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        state_topic = f'/{self.robot_ns}/robot_state' if self.robot_ns else '/robot_state'
        self.state_pub = self.create_publisher(String, state_topic, 10)

        # --- Timing & Velocity Tracking ---
        self.min_state_duration = 0.5  
        self.last_state_change_time = None
        self.last_tf_success_time = None
        self.tf_timeout_duration = 2.0 
        
        self.tactical_override_active = False
        self.last_enemy_dist = None
        self.last_eval_time = None

        # 30Hz Control Loop for instant state switching
        self.timer = self.create_timer(0.033, self.control_loop)
        self.get_logger().info("Smart State Machine Initialized (30Hz, Predictive Threat Active).")

    def get_pursuer_in_enemy_frame(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.enemy_frame, self.pursuer_frame, rclpy.time.Time()
            )
            return trans.transform.translation.x, trans.transform.translation.y
        except tf2_ros.TransformException:
            return None, None

    def evaluate_threat_state(self, rel_x: float, rel_y: float, dt: float):
        """Calculates danger using distance, angle, and approach velocity."""
        distance = math.hypot(rel_x, rel_y)
        relative_angle_deg = abs(math.degrees(math.atan2(rel_y, rel_x)))

        # Base Thresholds based on angle
        if relative_angle_deg <= 90.0:
            ratio = relative_angle_deg / 90.0
            base_threshold = 2.6 - (1.1 * ratio)
        else:
            ratio = (relative_angle_deg - 90.0) / 90.0
            base_threshold = 1.5 - (1.5 * ratio)

        # Predictive Velocity Expansion
        approach_speed = 0.0
        if self.last_enemy_dist is not None and dt > 0.0:
            approach_speed = (self.last_enemy_dist - distance) / dt
        self.last_enemy_dist = distance

        current_threshold = base_threshold
        # If enemy is charging fast (>0.2 m/s), expand the danger zone early
        if approach_speed > 0.2:
            predictive_buffer = min(0.8, approach_speed * 0.5) 
            current_threshold += predictive_buffer
        # If enemy is retreating, shrink danger zone so we resume chasing faster
        elif approach_speed < -0.2:
            current_threshold -= 0.2

        return distance <= current_threshold, distance, current_threshold

    def control_loop(self):
        now = self.get_clock().now()
        
        if self.last_state_change_time is None:
            self.last_state_change_time = now
            self.last_tf_success_time = now
            self.last_eval_time = now.nanoseconds / 1e9

        current_time_sec = now.nanoseconds / 1e9
        dt = current_time_sec - self.last_eval_time
        self.last_eval_time = current_time_sec
        time_in_state = current_time_sec - (self.last_state_change_time.nanoseconds / 1e9)

        rel_x, rel_y = self.get_pursuer_in_enemy_frame()
        
        # Dead Man's Switch: Revert to chasing if sensor is lost
        if rel_x is None: 
            if current_time_sec - (self.last_tf_success_time.nanoseconds / 1e9) > self.tf_timeout_duration and self.state != 'CHASING':
                self.state = 'CHASING'
                self.last_state_change_time = now
                self.last_enemy_dist = None  
            msg = String()
            msg.data = self.state
            self.state_pub.publish(msg)
            return

        self.last_tf_success_time = now

        in_danger, dist, threshold = self.evaluate_threat_state(rel_x, rel_y, dt)

        # State Transition Logic
        if self.state == 'CHASING':
            if in_danger:
                self.get_logger().warn(f'DANGER ZONE! Dist: {dist:.2f}m <= {threshold:.2f}m. EVADING.')
                self.state = 'EVADING'
                self.last_state_change_time = now

        elif self.state == 'EVADING':
            rel_angle_deg = abs(math.degrees(math.atan2(rel_y, rel_x)))
            
            # Angular Hysteresis: Hold state if we are behind the enemy
            if rel_angle_deg > 120.0:
                self.tactical_override_active = True
            elif rel_angle_deg < 110.0:
                self.tactical_override_active = False
            
            clear_of_threat = not in_danger and dist > (threshold + 0.2)
            timer_met = time_in_state >= self.min_state_duration
            
            if clear_of_threat and (timer_met or self.tactical_override_active):
                self.get_logger().info('Clear of threat. Resuming CHASING.')
                self.state = 'CHASING'
                self.last_state_change_time = now
                self.tactical_override_active = False

        msg = String()
        msg.data = self.state
        self.state_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(StateMachine())
    rclpy.shutdown()

if __name__ == '__main__':
    main()