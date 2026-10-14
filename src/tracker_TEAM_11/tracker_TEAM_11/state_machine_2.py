#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import tf2_ros

class StateMachine(Node):
    def __init__(self):
        super().__init__('state_machine')

        # --- Dynamic Namespace Configuration ---
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

        # --- TF2 Setup ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        state_topic = f'/{self.robot_ns}/robot_state' if self.robot_ns else '/robot_state'
        self.state_pub = self.create_publisher(String, state_topic, 10)

        # --- State Timers & Variables ---
        self.min_state_duration = 0.5  # Slightly reduced for faster switching
        self.last_state_change_time = None
        self.last_tf_success_time = None
        self.tf_timeout_duration = 2.0 
        
        self.tactical_override_active = False

        # --- SMART: Velocity Tracking Variables ---
        self.last_enemy_dist = None
        self.last_eval_time = None

        # --- FAST: 30Hz Control Loop ---
        # Upgraded from 0.1 (10Hz) to 0.033 (30Hz) for lightning-fast state transitions
        self.timer = self.create_timer(0.033, self.control_loop)
        self.get_logger().info("Smart State Machine Initialized (30Hz, Predictive Threat Active).")

    def get_pursuer_in_enemy_frame(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.enemy_frame, self.pursuer_frame, 
                rclpy.time.Time()
            )
            return trans.transform.translation.x, trans.transform.translation.y
        except tf2_ros.TransformException:
            return None, None

    def evaluate_threat_state(self, rel_x: float, rel_y: float, dt: float):
        """
        Determines danger based on distance, relative angle, AND the rate of closure.
        """
        distance = math.hypot(rel_x, rel_y)
        relative_angle_deg = abs(math.degrees(math.atan2(rel_y, rel_x)))

        # 1. Base Threshold (Static Profile)
        if relative_angle_deg <= 90.0:
            ratio = relative_angle_deg / 90.0
            base_threshold = 2.6 - (1.1 * ratio)
        else:
            ratio = (relative_angle_deg - 90.0) / 90.0
            base_threshold = 1.5 - (1.5 * ratio)

        # 2. SMART: Predictive Velocity Expansion
        # Calculate how fast the distance between bots is changing (m/s)
        approach_speed = 0.0
        if self.last_enemy_dist is not None and dt > 0.0:
            # Positive speed means they are getting closer; negative means moving away
            approach_speed = (self.last_enemy_dist - distance) / dt
        
        self.last_enemy_dist = distance

        # 3. Dynamic Threshold Adjustment
        current_threshold = base_threshold
        if approach_speed > 0.2:
            # Enemy is charging: Expand the danger zone predictively (up to 0.8m extra)
            # This triggers the evasion earlier if the enemy is fast.
            predictive_buffer = min(0.8, approach_speed * 0.5) 
            current_threshold += predictive_buffer
        elif approach_speed < -0.2:
            # Enemy is retreating: Shrink the danger zone slightly to resume chasing faster
            current_threshold -= 0.2

        if distance > current_threshold:
            return False, distance, current_threshold
            
        return True, distance, current_threshold

    def control_loop(self):
        now = self.get_clock().now()
        
        # Initialize timing variables on first run
        if self.last_state_change_time is None:
            self.last_state_change_time = now
            self.last_tf_success_time = now
            self.last_eval_time = now.nanoseconds / 1e9

        current_time_sec = now.nanoseconds / 1e9
        dt = current_time_sec - self.last_eval_time
        self.last_eval_time = current_time_sec
        time_in_state = current_time_sec - (self.last_state_change_time.nanoseconds / 1e9)

        rel_x, rel_y = self.get_pursuer_in_enemy_frame()
        
        # Dead Man's Switch: Sensor Loss Handling
        if rel_x is None: 
            time_since_last_tf = current_time_sec - (self.last_tf_success_time.nanoseconds / 1e9)
            
            if time_since_last_tf > self.tf_timeout_duration and self.state != 'CHASING':
                self.get_logger().warn(f'Enemy tracking lost for {time_since_last_tf:.1f}s. Reverting to CHASING.')
                self.state = 'CHASING'
                self.last_state_change_time = now
                self.last_enemy_dist = None  # Reset velocity tracker
            
            msg = String()
            msg.data = self.state
            self.state_pub.publish(msg)
            return

        self.last_tf_success_time = now

        # Evaluate threat using distance, angle, and time delta (dt) for velocity
        in_danger, dist, threshold = self.evaluate_threat_state(rel_x, rel_y, dt)

        # --- STATE TRANSITION LOGIC ---
        if self.state == 'CHASING':
            if in_danger:
                self.get_logger().warn(f'DANGER ZONE! Dist: {dist:.2f}m <= {threshold:.2f}m. EVADING.')
                self.state = 'EVADING'
                self.last_state_change_time = now

        elif self.state == 'EVADING':
            rel_angle_deg = abs(math.degrees(math.atan2(rel_y, rel_x)))
            
            # Angular Hysteresis for Tactical Override
            if rel_angle_deg > 120.0:
                self.tactical_override_active = True
            elif rel_angle_deg < 110.0:
                self.tactical_override_active = False
            
            # Clearance requires stepping 0.2m outside the dynamic threshold
            clear_of_threat = not in_danger and dist > (threshold + 0.2)
            timer_met = time_in_state >= self.min_state_duration
            
            # Exit evasion if we are clear AND (we waited long enough OR secured the enemy's rear)
            if clear_of_threat and (timer_met or self.tactical_override_active):
                reason = "Rear Advantage" if self.tactical_override_active else "Timer Met"
                self.get_logger().info(f'Clear of threat ({reason}). Resuming CHASING.')
                self.state = 'CHASING'
                self.last_state_change_time = now
                self.tactical_override_active = False

        # Broadcast State
        msg = String()
        msg.data = self.state
        self.state_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(StateMachine())
    rclpy.shutdown()

if __name__ == '__main__':
    main()