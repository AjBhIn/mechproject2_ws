# state_machine_2.py
#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import tf2_ros

# --- GLOBAL TUNING VARIABLES ---
CONTROL_LOOP_RATE_SEC = 0.033

# Evaluates danger differently based on where we are relative to the enemy. 
# Front approach triggers evasion far away (1.2m), rear approach allows getting close (0.25m).
BASE_DANGER_THRESHOLD_FRONT = 1.2  
BASE_DANGER_THRESHOLD_REAR = 0.25   

# --- EVASION LOCK VARIABLES ---
REPOSITION_CLEAR_DIST = 2.0        # Must physically achieve 2.0m distance to end evasion
REPOSITION_CLEAR_ANGLE = 120.0     # Must secure an angle >= 120deg (rear pocket) to end evasion

# Predictive velocity buffers
FAST_APPROACH_SPEED = 0.2
FAST_RETREAT_SPEED = -0.2
MAX_PREDICTIVE_BUFFER = 0.8

# Offensive Commit Zones
STRIKE_ZONE_DIST = 1.2             
STRIKE_ZONE_ANGLE_DEG = 135.0      
STRIKE_ZONE_BUFFER_DIST = 0.3      
STRIKE_ZONE_BUFFER_ANGLE = 15.0    
CAPTURE_DISTANCE = 0.5             
COOLDOWN_DURATION_SEC = 5.0        

TF_TIMEOUT_SEC = 2.5               

# [OPTIMIZED]: Global debounce. Prevents the state machine from instantly 
# fluttering between CHASING and EVADING due to tiny sensor noise at the boundary.
MIN_STATE_LOCK_SEC = 0.5

class StateMachine(Node):
    """
    High-level operational brain. Analyzes real-time geometry between the robot and enemy,
    using hysteresis and velocity prediction to transition smoothly between CHASING, 
    STRIKING, EVADING, and COOLDOWN states.
    """
    def __init__(self):
        super().__init__('state_machine')

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

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        state_topic = f'/{self.robot_ns}/robot_state' if self.robot_ns else '/robot_state'
        self.state_pub = self.create_publisher(String, state_topic, 10)

        # Temporal trackers
        self.last_state_change_time = None
        self.last_tf_success_time = None
        self.last_enemy_dist = None
        self.last_eval_time = None

        self.timer = self.create_timer(CONTROL_LOOP_RATE_SEC, self.control_loop)
        self.get_logger().info(f"[STATE MACHINE] Initialized. Anti-Flutter Hysteresis Lock ({MIN_STATE_LOCK_SEC}s) Active.")

    def get_pursuer_in_enemy_frame(self):
        """Calculates our precise Cartesian location localized entirely to the enemy's POV."""
        try:
            trans = self.tf_buffer.lookup_transform(
                self.enemy_frame, self.pursuer_frame, rclpy.time.Time()
            )
            return trans.transform.translation.x, trans.transform.translation.y
        except tf2_ros.TransformException:
            return None, None

    def evaluate_threat_state(self, rel_x: float, rel_y: float, dt: float):
        """
        Calculates the "in_danger" boolean by projecting a dynamic threat-radius bubble.
        The bubble stretches far in front of the enemy (1.2m) but shrinks to (0.25m) at the rear.
        Additionally inflates the bubble if the enemy is accelerating rapidly toward us.
        """
        distance = math.hypot(rel_x, rel_y)
        relative_angle_deg = abs(math.degrees(math.atan2(rel_y, rel_x)))

        # Morphing the shape of the danger threshold based on approach angle
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
        # Predictive inflation: Expand the threshold if the enemy is diving at us fast
        if approach_speed > FAST_APPROACH_SPEED:
            predictive_buffer = min(MAX_PREDICTIVE_BUFFER, approach_speed * 0.5) 
            current_threshold += predictive_buffer
        # Predictive deflation: If enemy is driving away, we can get slightly closer safely
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

        rel_x, rel_y = self.get_pursuer_in_enemy_frame()
        
        # TF Drop protection: Force default state if data stops flowing
        if rel_x is None: 
            if current_time_sec - (self.last_tf_success_time.nanoseconds / 1e9) > TF_TIMEOUT_SEC and self.state != 'CHASING':
                self.get_logger().error(f"[STATE MACHINE] TF Data lost. Aborting {self.state} and defaulting to CHASING.")
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
        time_in_state = current_time_sec - (self.last_state_change_time.nanoseconds / 1e9)

        # Hysteresis check ensures non-emergency state shifts don't jitter
        can_transition = (time_in_state >= MIN_STATE_LOCK_SEC)

        # --- STATE MACHINE EVALUATION LOGIC ---
        if self.state == 'COOLDOWN':
            # Danger overrides the lock buffer to keep the bot safe
            if in_danger:
                self.get_logger().warn(f'[STATE MACHINE] Under attack during Cooldown! Forcing EVADING.')
                self.state = 'EVADING'
                self.last_state_change_time = now
            elif time_in_state >= COOLDOWN_DURATION_SEC:
                self.get_logger().info('[STATE MACHINE] Cooldown complete. Resuming offensive CHASING.')
                self.state = 'CHASING'
                self.last_state_change_time = now

        elif self.state == 'CHASING':
            if in_danger:
                self.get_logger().warn(f'[STATE MACHINE] Physical Threat Triggered (Dist: {dist:.2f}m). Switching to EVADING.')
                self.state = 'EVADING'
                self.last_state_change_time = now
            elif dist <= STRIKE_ZONE_DIST and rel_angle_deg >= STRIKE_ZONE_ANGLE_DEG and can_transition:
                self.get_logger().info(f'[STATE MACHINE] Rear Pocket Secured (Angle: {rel_angle_deg:.1f}deg). Committing to STRIKING!')
                self.state = 'STRIKING'
                self.last_state_change_time = now

        elif self.state == 'STRIKING':
            if dist <= CAPTURE_DISTANCE:
                self.get_logger().info(f'[STATE MACHINE] >> SUCCESSFUL CAPTURE << Halting offense for COOLDOWN ({COOLDOWN_DURATION_SEC}s).')
                self.state = 'COOLDOWN'
                self.last_state_change_time = now
            elif in_danger:
                self.get_logger().warn(f'[STATE MACHINE] Enemy countered during Strike. Breaking off to EVADING.')
                self.state = 'EVADING'
                self.last_state_change_time = now
            # Abort Strike if enemy slips out of the extended spatial buffer
            elif (dist > (STRIKE_ZONE_DIST + STRIKE_ZONE_BUFFER_DIST) or rel_angle_deg < (STRIKE_ZONE_ANGLE_DEG - STRIKE_ZONE_BUFFER_ANGLE)) and can_transition:
                self.get_logger().info('[STATE MACHINE] Enemy broke pocket lock. Aborting Strike -> Reverting to CHASING.')
                self.state = 'CHASING'
                self.last_state_change_time = now

        elif self.state == 'EVADING':
            # Evading lock requires massive distance, a safe rear-angle location, AND time duration to conclude.
            is_safely_behind = (rel_angle_deg >= REPOSITION_CLEAR_ANGLE)
            is_at_safe_distance = (dist >= REPOSITION_CLEAR_DIST)
            
            # Print evaluation criteria cleanly once per second to prevent log spam
            if int(now.nanoseconds / 1e9 * 30) % 30 == 0:
                self.get_logger().info(f"[STATE MACHINE] Evasion Status - Distance: {dist:.2f}m (Target {REPOSITION_CLEAR_DIST}m) | Angle: {rel_angle_deg:.1f}° (Target {REPOSITION_CLEAR_ANGLE}°)")
            
            if is_safely_behind and is_at_safe_distance and can_transition:
                self.get_logger().info(f'[STATE MACHINE] Evasion Checkpoints Cleared. Distance & Angle secured. Resuming CHASING.')
                self.state = 'CHASING'
                self.last_state_change_time = now

        # Broadcast the verified state so calculator nodes can listen
        msg = String()
        msg.data = self.state
        self.state_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(StateMachine())
    rclpy.shutdown()

if __name__ == '__main__':
    main()