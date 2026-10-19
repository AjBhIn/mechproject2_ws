#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import tf2_ros

# --- GLOBAL TUNING VARIABLES ---
CONTROL_LOOP_RATE_SEC = 0.033

# Evaluates danger differently based on where we are relative to the enemy. 
BASE_DANGER_THRESHOLD_FRONT = 1.2  
BASE_DANGER_THRESHOLD_REAR = 0.25   

# --- EVASION LOCK VARIABLES ---
# [OPTIMIZED]: Swapped degrees for native radians, bypassing math.radians() in execution
REPOSITION_CLEAR_DIST = 2.0        
REPOSITION_CLEAR_ANGLE_RAD = math.radians(120.0)

FAST_APPROACH_SPEED = 0.2
FAST_RETREAT_SPEED = -0.2
MAX_PREDICTIVE_BUFFER = 0.8

STRIKE_ZONE_DIST = 1.2             
STRIKE_ZONE_ANGLE_RAD = math.radians(135.0)      
STRIKE_ZONE_BUFFER_DIST = 0.3      
STRIKE_ZONE_BUFFER_ANGLE_RAD = math.radians(15.0)
CAPTURE_DISTANCE = 0.5 # IF the bot and enemy distance is 50cm or less the, this is capture distnace       
COOLDOWN_DURATION_SEC = 5.0        

TF_TIMEOUT_SEC = 2.5               
MIN_STATE_LOCK_SEC = 0.5

class StateMachine(Node):
    """
    High-level operational brain. 
    [OPTIMIZED]: Zero-Conversion 
    Native Radian Tracking.
    """
    def __init__(self):
        super().__init__('state_machine')

        # Dynamic Namespace 
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

        # Default State
        self.state = 'CHASING'

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Robot state publisher: Publishes state of the robot
        state_topic = f'/{self.robot_ns}/robot_state' if self.robot_ns else '/robot_state'
        self.state_pub = self.create_publisher(String, state_topic, 10)

        self.last_state_change_time = None
        self.last_tf_success_time = None
        self.last_enemy_dist = None
        self.last_eval_time = None

        self.timer = self.create_timer(CONTROL_LOOP_RATE_SEC, self.control_loop)
        self.get_logger().info(f"[STATE MACHINE] Initialized. Native Radian Geometry Active.")

    # Getting baseline of our bot from the enemy perspective
    def get_pursuer_in_enemy_frame(self):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.enemy_frame, self.pursuer_frame, rclpy.time.Time()
            )
            return trans.transform.translation.x, trans.transform.translation.y
        except tf2_ros.TransformException:
            return None, None

    # Finding out our we in evading: just that
    def evaluate_threat_state(self, rel_x: float, rel_y: float, dt: float):
        distance = math.hypot(rel_x, rel_y)
        relative_angle_rad = abs(math.atan2(rel_y, rel_x))

        # Native radian ratio morphing avoids repetitive math.degrees() calls
        # If the front facing angle is less then 180deg, then this
        if relative_angle_rad <= (math.pi / 2.0):
            ratio = relative_angle_rad / (math.pi / 2.0)
            base_threshold = BASE_DANGER_THRESHOLD_FRONT - ((BASE_DANGER_THRESHOLD_FRONT - BASE_DANGER_THRESHOLD_REAR) * ratio)
        else:
            ratio = (relative_angle_rad - (math.pi / 2.0)) / (math.pi / 2.0)
            base_threshold = BASE_DANGER_THRESHOLD_REAR - (BASE_DANGER_THRESHOLD_REAR * ratio)

        approach_speed = 0.0
        if self.last_enemy_dist is not None and dt > 0.0:
            approach_speed = (self.last_enemy_dist - distance) / dt
        self.last_enemy_dist = distance

        # A buffer added to our threat
        current_threshold = base_threshold
        if approach_speed > FAST_APPROACH_SPEED:
            predictive_buffer = min(MAX_PREDICTIVE_BUFFER, approach_speed * 0.5) 
            current_threshold += predictive_buffer
        elif approach_speed < FAST_RETREAT_SPEED:
            current_threshold -= 0.2

        # Sending bool, distance, threshold
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

        # Getting our enemy tf
        rel_x, rel_y = self.get_pursuer_in_enemy_frame()

        # Checking if lost tf tree
        if rel_x is None: 
            if (current_time_sec - (self.last_tf_success_time.nanoseconds / 1e9) > TF_TIMEOUT_SEC) and (self.state != 'CHASING'):
                self.get_logger().error(f"[STATE MACHINE] TF Data lost. Aborting {self.state} and defaulting to CHASING.")
                self.state = 'CHASING'
                self.last_state_change_time = now
                self.last_enemy_dist = None  
            msg = String()
            msg.data = self.state
            self.state_pub.publish(msg)
            return

        self.last_tf_success_time = now

        # Checking Threat
        # Passing in our enemy distance
        in_danger, dist, threshold = self.evaluate_threat_state(rel_x, rel_y, dt)
        rel_angle_rad = abs(math.atan2(rel_y, rel_x)) # Getting the angle of enemy with respect our bot
        time_in_state = current_time_sec - (self.last_state_change_time.nanoseconds / 1e9)

        can_transition = (time_in_state >= MIN_STATE_LOCK_SEC)

        if self.state == 'COOLDOWN':
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
            elif dist <= STRIKE_ZONE_DIST and rel_angle_rad >= STRIKE_ZONE_ANGLE_RAD and can_transition:
                self.get_logger().info(f'[STATE MACHINE] Rear Pocket Secured (Angle: {math.degrees(rel_angle_rad):.1f}deg). Committing to STRIKING!')
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
            elif (dist > (STRIKE_ZONE_DIST + STRIKE_ZONE_BUFFER_DIST) or rel_angle_rad < (STRIKE_ZONE_ANGLE_RAD - STRIKE_ZONE_BUFFER_ANGLE_RAD)) and can_transition:
                self.get_logger().info('[STATE MACHINE] Enemy broke pocket lock. Aborting Strike -> Reverting to CHASING.')
                self.state = 'CHASING'
                self.last_state_change_time = now

        elif self.state == 'EVADING':
            is_safely_behind = (rel_angle_rad >= REPOSITION_CLEAR_ANGLE_RAD)
            is_at_safe_distance = (dist >= REPOSITION_CLEAR_DIST)
            
            if int(now.nanoseconds / 1e9 * 30) % 30 == 0:
                self.get_logger().info(f"[STATE MACHINE] Evasion Status - Distance: {dist:.2f}m (Target {REPOSITION_CLEAR_DIST}m) | Angle: {math.degrees(rel_angle_rad):.1f}° (Target {math.degrees(REPOSITION_CLEAR_ANGLE_RAD):.1f}°)")
            
            if is_safely_behind and is_at_safe_distance and can_transition:
                self.get_logger().info(f'[STATE MACHINE] Evasion Checkpoints Cleared. Distance & Angle secured. Resuming CHASING.')
                self.state = 'CHASING'
                self.last_state_change_time = now

        # Publishing to our robot_state topic
        msg = String()
        msg.data = self.state
        self.state_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(StateMachine())
    rclpy.shutdown()

if __name__ == '__main__':
    main()