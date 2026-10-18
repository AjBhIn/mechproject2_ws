#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import tf2_ros

# --- GLOBAL TUNING VARIABLES ---
CONTROL_LOOP_RATE_SEC = 0.033

BASE_DANGER_THRESHOLD_FRONT = 1.2  
BASE_DANGER_THRESHOLD_REAR = 0.25   

# --- EVASION LOCK VARIABLES ---
REPOSITION_CLEAR_DIST = 2.0        # Must be at least 2.0m away
REPOSITION_CLEAR_ANGLE = 120.0     # Must be securely behind the enemy's front view

FAST_APPROACH_SPEED = 0.2
FAST_RETREAT_SPEED = -0.2
MAX_PREDICTIVE_BUFFER = 0.8

STRIKE_ZONE_DIST = 1.2             
STRIKE_ZONE_ANGLE_DEG = 135.0      
STRIKE_ZONE_BUFFER_DIST = 0.3      
STRIKE_ZONE_BUFFER_ANGLE = 15.0    
CAPTURE_DISTANCE = 0.5             
COOLDOWN_DURATION_SEC = 5.0        

TF_TIMEOUT_SEC = 2.5               

class StateMachine(Node):
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

        self.last_state_change_time = None
        self.last_tf_success_time = None
        self.last_enemy_dist = None
        self.last_eval_time = None

        self.timer = self.create_timer(CONTROL_LOOP_RATE_SEC, self.control_loop)
        self.get_logger().info("[STATE MACHINE] Initialized. Strict 2m Rear Repositioning Active.")

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

        rel_x, rel_y = self.get_pursuer_in_enemy_frame()
        
        if rel_x is None: 
            if current_time_sec - (self.last_tf_success_time.nanoseconds / 1e9) > TF_TIMEOUT_SEC and self.state != 'CHASING':
                self.get_logger().warn(f"[STATE MACHINE] TF Data lost. Reverting to CHASING.")
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

        if self.state == 'COOLDOWN':
            if in_danger:
                self.get_logger().warn(f'[STATE MACHINE] Threatened! Breaking Cooldown -> EVADING.')
                self.state = 'EVADING'
                self.last_state_change_time = now
            elif time_in_state >= COOLDOWN_DURATION_SEC:
                self.get_logger().info('[STATE MACHINE] Cooldown complete. Resuming CHASING.')
                self.state = 'CHASING'
                self.last_state_change_time = now

        elif self.state == 'CHASING':
            if in_danger:
                self.get_logger().warn(f'[STATE MACHINE] Danger detected. Switching -> EVADING.')
                self.state = 'EVADING'
                self.last_state_change_time = now
            elif dist <= STRIKE_ZONE_DIST and rel_angle_deg >= STRIKE_ZONE_ANGLE_DEG:
                self.get_logger().info(f'[STATE MACHINE] In Strike Zone. Committing to STRIKING!')
                self.state = 'STRIKING'
                self.last_state_change_time = now

        elif self.state == 'STRIKING':
            if dist <= CAPTURE_DISTANCE:
                self.get_logger().info(f'[STATE MACHINE] Target captured! Entering COOLDOWN ({COOLDOWN_DURATION_SEC}s).')
                self.state = 'COOLDOWN'
                self.last_state_change_time = now
            elif in_danger:
                self.get_logger().warn(f'[STATE MACHINE] Counterattack during strike -> EVADING.')
                self.state = 'EVADING'
                self.last_state_change_time = now
            elif dist > (STRIKE_ZONE_DIST + STRIKE_ZONE_BUFFER_DIST) or rel_angle_deg < (STRIKE_ZONE_ANGLE_DEG - STRIKE_ZONE_BUFFER_ANGLE):
                self.get_logger().info('[STATE MACHINE] Enemy pulled away. Aborting strike -> CHASING.')
                self.state = 'CHASING'
                self.last_state_change_time = now

        elif self.state == 'EVADING':
            is_safely_behind = (rel_angle_deg >= REPOSITION_CLEAR_ANGLE)
            is_at_safe_distance = (dist >= REPOSITION_CLEAR_DIST)
            
            # Print evaluation criteria occasionally for debugging
            if int(now.nanoseconds / 1e9 * 30) % 30 == 0:  # Logs ~1 time per second
                self.get_logger().info(f"[STATE MACHINE] Evaluating Evasion - Dist: {dist:.2f}m (Need {REPOSITION_CLEAR_DIST}m), Angle: {rel_angle_deg:.1f}deg (Need {REPOSITION_CLEAR_ANGLE}deg)")
            
            if is_safely_behind and is_at_safe_distance:
                self.get_logger().info(f'[STATE MACHINE] Successfully cleared 2.0m behind enemy. Resuming CHASING.')
                self.state = 'CHASING'
                self.last_state_change_time = now

        msg = String()
        msg.data = self.state
        self.state_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(StateMachine())
    rclpy.shutdown()

if __name__ == '__main__':
    main()