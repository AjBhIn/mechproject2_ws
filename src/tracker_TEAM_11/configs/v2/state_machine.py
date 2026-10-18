#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import tf2_ros

# --- GLOBAL TUNING VARIABLES ---
# Loop rate for the state machine (30Hz)
CONTROL_LOOP_RATE_SEC = 0.033

# Evasion distance thresholds
BASE_DANGER_THRESHOLD_FRONT = 1.2  # High danger if enemy faces us
# LOWERED to 0.4m so we can safely hold the 0.5m capture pocket without triggering evasion
BASE_DANGER_THRESHOLD_REAR = 0.4   

# Approach speed thresholds for predictive evasion (m/s)
FAST_APPROACH_SPEED = 0.2
FAST_RETREAT_SPEED = -0.2
MAX_PREDICTIVE_BUFFER = 0.8

# Angular hysteresis to prevent flickering
HYSTERESIS_ANGLE_HIGH = 120.0
HYSTERESIS_ANGLE_LOW = 110.0

# Strike & Cooldown Variables
STRIKE_ZONE_DIST = 1.2             # Max distance to initiate the STRIKING dive
STRIKE_ZONE_ANGLE_DEG = 135.0      # Must be behind the enemy (180 deg is perfectly behind)
CAPTURE_DISTANCE = 0.5             # Distance at which the photo is taken and cooldown begins
COOLDOWN_DURATION_SEC = 5.0        # How long to wait before next photo

# Timeouts
MIN_STATE_DURATION_SEC = 0.5       
TF_TIMEOUT_SEC = 2.5               

class StateMachine(Node):
    """
    Velocity-Aware Multi-Phase State Machine. 
    Handles CHASING, STRIKING, COOLDOWN, and EVADING states.
    Allows holding the pocket during cooldown but breaks to evade if threatened.
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

        # --- Timing & Tracking ---
        self.last_state_change_time = None
        self.last_tf_success_time = None
        
        self.tactical_override_active = False
        self.last_enemy_dist = None
        self.last_eval_time = None

        self.timer = self.create_timer(CONTROL_LOOP_RATE_SEC, self.control_loop)
        self.get_logger().info("Multi-Phase State Machine Initialized (Interruptible Cooldown Active).")

    def get_pursuer_in_enemy_frame(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.enemy_frame, self.pursuer_frame, rclpy.time.Time()
            )
            return trans.transform.translation.x, trans.transform.translation.y
        except tf2_ros.TransformException:
            return None, None

    def evaluate_threat_state(self, rel_x: float, rel_y: float, dt: float):
        distance = math.hypot(rel_x, rel_y)
        relative_angle_deg = abs(math.degrees(math.atan2(rel_y, rel_x)))

        if relative_angle_deg <= 90.0:
            ratio = relative_angle_deg / 90.0
            base_threshold = BASE_DANGER_THRESHOLD_FRONT - ((BASE_DANGER_THRESHOLD_FRONT - BASE_DANGER_THRESHOLD_REAR) * ratio)
        else:
            ratio = (relative_angle_deg - 90.0) / 90.0
            base_threshold = BASE_DANGER_THRESHOLD_REAR - (BASE_DANGER_THRESHOLD_REAR * ratio)

        approach_speed = 0.0
        if self.last_enemy_dist is not None and dt > 0.0:
            approach_speed = (self.last_enemy_dist - distance) / dt
        self.last_enemy_dist = distance

        current_threshold = base_threshold
        if approach_speed > FAST_APPROACH_SPEED:
            predictive_buffer = min(MAX_PREDICTIVE_BUFFER, approach_speed * 0.5) 
            current_threshold += predictive_buffer
        elif approach_speed < FAST_RETREAT_SPEED:
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
        
        if rel_x is None: 
            if current_time_sec - (self.last_tf_success_time.nanoseconds / 1e9) > TF_TIMEOUT_SEC and self.state != 'CHASING':
                self.get_logger().warn(f"TF Data lost. Reverting to CHASING.")
                self.state = 'CHASING'
                self.last_state_change_time = now
                self.last_enemy_dist = None  
            msg = String()
            msg.data = self.state
            self.state_pub.publish(msg)
            return

        self.last_tf_success_time = now

        in_danger, dist, threshold = self.evaluate_threat_state(rel_x, rel_y, dt)
        rel_angle_deg = abs(math.degrees(math.atan2(rel_y, rel_x)))

        # --- STATE TRANSITION LOGIC ---

        if self.state == 'COOLDOWN':
            # Emergency Break: If the enemy turns to attack, break cooldown and evade
            if in_danger:
                self.get_logger().warn(f'Enemy turned to attack during cooldown! EVADING.')
                self.state = 'EVADING'
                self.last_state_change_time = now
            # Otherwise, hold the pocket until the timer clears
            elif time_in_state >= COOLDOWN_DURATION_SEC:
                self.get_logger().info('Cooldown complete. Resuming CHASING/STRIKING.')
                self.state = 'CHASING'
                self.last_state_change_time = now

        elif self.state == 'CHASING':
            if in_danger:
                self.get_logger().warn(f'Danger detected. Switching to EVADING.')
                self.state = 'EVADING'
                self.last_state_change_time = now
            elif dist <= STRIKE_ZONE_DIST and rel_angle_deg >= STRIKE_ZONE_ANGLE_DEG:
                self.get_logger().info(f'In Strike Zone (Dist: {dist:.2f}m, Angle: {rel_angle_deg:.0f}deg). STRIKING!')
                self.state = 'STRIKING'
                self.last_state_change_time = now

        elif self.state == 'STRIKING':
            if dist <= CAPTURE_DISTANCE:
                self.get_logger().info(f'Target captured! Entering COOLDOWN for {COOLDOWN_DURATION_SEC}s.')
                self.state = 'COOLDOWN'
                self.last_state_change_time = now
            elif in_danger:
                self.get_logger().warn(f'Enemy engaged during strike. EVADING.')
                self.state = 'EVADING'
                self.last_state_change_time = now
            elif dist > STRIKE_ZONE_DIST + 0.3 or rel_angle_deg < STRIKE_ZONE_ANGLE_DEG - 15.0:
                self.get_logger().info('Lost strike position. Resuming CHASING setup.')
                self.state = 'CHASING'
                self.last_state_change_time = now

        elif self.state == 'EVADING':
            if rel_angle_deg > HYSTERESIS_ANGLE_HIGH:
                self.tactical_override_active = True
            elif rel_angle_deg < HYSTERESIS_ANGLE_LOW:
                self.tactical_override_active = False
            
            clear_of_threat = not in_danger and dist > (threshold + 0.2)
            timer_met = time_in_state >= MIN_STATE_DURATION_SEC
            
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