#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
import tf2_ros
from geometry_msgs.msg import TransformStamped
import numpy as np

# --- GLOBAL TUNING VARIABLES ---
LOOP_RATE_SEC = 0.033

COSTMAP_MARGIN_CELLS = 6
MAX_SAFE_COST = 85
MAX_EMERGENCY_COST = 95
STUCK_ESCAPE_DIST = 1.2

CAPTURE_ORIENT_DIST = 0.35
MIN_YAW_UPDATE_DIST = 0.15
DIST_STEPS = [0.2, 0.4, 0.6, 0.8]
STUCK_DIST_STEPS = [1.2, 1.0, 0.8, 0.6]

CHASE_LEASH_DIST = 0.8
STRIKE_TARGET_DIST = 0.10
ORBITAL_FLANK_DIST = 1.2

MAX_YAW_RATE_SMOOTHING = 1.0
LOG_THROTTLE_SEC = 1.0

class ChaserCalculator(Node):
    """
    Calculates the offensive pursuit trajectory for the robot. 
    [OPTIMIZED]: Zero-allocation math. All trig grids pre-computed.
    """
    def __init__(self):
        super().__init__('chaser_calculator')

        # Dynamic Namespace
        raw_ns = self.get_namespace().strip('/')
        default_robot_ns = raw_ns if raw_ns else 'our_bot'

        self.declare_parameter('chaser', default_robot_ns)
        self.declare_parameter('enemy', 'enemy_bot')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('pursuer_frame', '')
        self.declare_parameter('enemy_frame', '')
        self.declare_parameter('carrot_frame', '')
        self.declare_parameter('broadcast_frame', 'chase_nav_target')

        self.robot_ns = self.get_parameter('chaser').value.strip('/')
        self.enemy_ns = self.get_parameter('enemy').value.strip('/')
        self.map_frame = self.get_parameter('map_frame').value
        
        pursuer_p = self.get_parameter('pursuer_frame').value
        self.pursuer_frame = pursuer_p if pursuer_p else (f'{self.robot_ns}/base_link' if self.robot_ns else 'base_link')

        enemy_p = self.get_parameter('enemy_frame').value
        self.enemy_frame = enemy_p if enemy_p else (f'{self.enemy_ns}/base_link' if self.enemy_ns else 'base_link')

        carrot_p = self.get_parameter('carrot_frame').value
        self.carrot_frame = carrot_p if carrot_p else (f'{self.enemy_ns}/target_point' if self.enemy_ns else 'target_point')
        self.broadcast_frame = self.get_parameter('broadcast_frame').value

        # --- PRE-COMPUTED MEMORY-SAFE CACHES ---
        # Moving np.linspace and trig functions out of the 30Hz loop kills garbage collection spikes
        self.probe_angles = np.linspace(0, 2 * math.pi, 36, endpoint=False)
        self.probe_cos = np.cos(self.probe_angles)
        self.probe_sin = np.sin(self.probe_angles)

        self.current_state = 'CHASING'
        self.costmap = None
        self.map_data = None
        self.map_res = 0.05
        self.map_width = self.map_height = 0
        self.map_origin_x = self.map_origin_y = 0.0

        self.last_valid_yaw = 0.0
        self.last_enemy_yaw = None
        self.last_enemy_time = None
        self.enemy_yaw_rate = 0.0
        self.filtered_target_x = None
        self.filtered_target_y = None

        # [OPTIMIZED]: Pre-computed target-centric arc offsets for fast pathing around obstacles
        arc_radii = np.array(DIST_STEPS)
        arc_angles = np.deg2rad(np.arange(0, 360, 30))
        r_grid, a_grid = np.meshgrid(arc_radii, arc_angles)
        self.arc_dx = (r_grid * np.cos(a_grid)).flatten()
        self.arc_dy = (r_grid * np.sin(a_grid)).flatten()
        self.arc_r = r_grid.flatten() # Pre-cached distance from candidate to true target

        self.flank_active_logged = False
        self.dive_active_logged = False
        self.was_sleeping = False
        self.last_log_time = 0.0
        self.flank_direction_bias = 1.0
        self.flank_lock_time = 0.0
        
        self.is_recovering = False
        self.high_cost_start_time = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        state_topic = f'/{self.robot_ns}/robot_state' if self.robot_ns else '/robot_state'
        costmap_topic = f'/{self.robot_ns}/global_costmap/costmap' if self.robot_ns else '/global_costmap/costmap'

        self.create_subscription(String, state_topic, self.state_cb, 10)
        self.create_subscription(OccupancyGrid, costmap_topic, self.costmap_cb, 10)

        self.timer = self.create_timer(LOOP_RATE_SEC, self.control_loop)
        self.get_logger().info("[CHASER CALCULATOR] Initialized. Zero-Allocation Math Arrays Active.")

    def state_cb(self, msg):
        if self.current_state != msg.data:
            self.get_logger().debug(f"[CHASER] State Transition: {self.current_state} -> {msg.data}")
        self.current_state = msg.data

    def costmap_cb(self, msg: OccupancyGrid):
        self.costmap = msg
        self.map_data = np.array(msg.data, dtype=np.int16)
        self.map_res = msg.info.resolution
        self.map_width, self.map_height = msg.info.width, msg.info.height
        self.map_origin_x, self.map_origin_y = msg.info.origin.position.x, msg.info.origin.position.y

    @staticmethod
    def get_yaw_from_quaternion(q):
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def get_cell_cost(self, x: float, y: float) -> int:
        if self.map_data is None: return 100
        raw_col = int((x - self.map_origin_x) / self.map_res)
        raw_row = int((y - self.map_origin_y) / self.map_res)

        if raw_col < COSTMAP_MARGIN_CELLS or raw_col >= (self.map_width - COSTMAP_MARGIN_CELLS) or \
           raw_row < COSTMAP_MARGIN_CELLS or raw_row >= (self.map_height - COSTMAP_MARGIN_CELLS):
            return 100

        col = max(0, min(raw_col, self.map_width - 1))
        row = max(0, min(raw_row, self.map_height - 1))
        cost = self.map_data[row * self.map_width + col]
        return 100 if cost == -1 else cost

    def get_cell_costs_vectorized(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        if self.map_data is None: return np.full(xs.shape, 100)
        
        raw_cols = ((xs - self.map_origin_x) / self.map_res).astype(np.int32)
        raw_rows = ((ys - self.map_origin_y) / self.map_res).astype(np.int32)
        
        invalid = (raw_cols < COSTMAP_MARGIN_CELLS) | (raw_cols >= (self.map_width - COSTMAP_MARGIN_CELLS)) | \
                  (raw_rows < COSTMAP_MARGIN_CELLS) | (raw_rows >= (self.map_height - COSTMAP_MARGIN_CELLS))
                  
        cols = np.clip(raw_cols, 0, self.map_width - 1)
        rows = np.clip(raw_rows, 0, self.map_height - 1)
        
        costs = self.map_data[rows * self.map_width + cols]
        costs = np.where((costs == -1) | invalid, 100, costs)
        return costs

    def is_pose_safe(self, x: float, y: float, max_cost: int = MAX_SAFE_COST) -> bool:
        if self.costmap is None: return True
        return self.get_cell_cost(x, y) <= max_cost

    # Recovery method when the bot is stuck in inflated zone.
    def get_chaser_stuck_recovery_target(self, our_x: float, our_y: float, our_yaw: float,
                                         enemy_x: float, enemy_y: float, carrot_x: float, carrot_y: float,
                                         current_cost: int, should_log: bool):
        # [OPTIMIZED]: Uses pre-allocated self.probe_cos and self.probe_sin arrays
        probe_x = our_x + 0.6 * self.probe_cos
        probe_y = our_y + 0.6 * self.probe_sin
        probe_costs = self.get_cell_costs_vectorized(probe_x, probe_y)

        cost_drop = current_cost - probe_costs
        scores = cost_drop * 15.0

        angle_to_enemy = math.atan2(enemy_y - our_y, enemy_x - our_x)
        enemy_diff = np.abs(np.arctan2(np.sin(self.probe_angles - angle_to_enemy), np.cos(self.probe_angles - angle_to_enemy)))
        scores = np.where(enemy_diff < np.deg2rad(75.0), scores - 500.0, scores)

        dist_to_carrot = np.hypot(probe_x - carrot_x, probe_y - carrot_y)
        scores -= dist_to_carrot * 5.0

        valid_angles = probe_costs < MAX_EMERGENCY_COST

        if should_log:
            debug_costs = [f"{int(math.degrees(ang)) % 360}°:{cost}" for ang, cost in zip(self.probe_angles, probe_costs)]
            self.get_logger().info(f"[CHASER STUCK SWEEP] Robot Center Cost: {current_cost} | Evaluated 360 Sweep.")

        if not np.any(valid_angles):
            self.get_logger().warn("[CHASER STUCK] Boxed in! Executing 90-degree lateral blind escape.")
            cross_product = math.cos(our_yaw) * math.sin(angle_to_enemy) - math.sin(our_yaw) * math.cos(angle_to_enemy)
            spin_dir = -1.0 if cross_product > 0 else 1.0
            best_escape_yaw = our_yaw + (spin_dir * math.pi / 2.0)
            
            spin_x = our_x + 0.6 * math.cos(best_escape_yaw)
            spin_y = our_y + 0.6 * math.sin(best_escape_yaw)
            return spin_x, spin_y, best_escape_yaw

        best_idx = np.argmax(np.where(valid_angles, scores, -1e9))
        best_escape_yaw = float(self.probe_angles[best_idx])

        escape_dist = 0.6
        for d in STUCK_DIST_STEPS:
            tx = our_x + d * math.cos(best_escape_yaw)
            ty = our_y + d * math.sin(best_escape_yaw)
            if self.get_cell_cost(tx, ty) < MAX_SAFE_COST and math.hypot(tx - enemy_x, ty - enemy_y) >= 0.45:
                escape_dist = d
                break

        rec_x = our_x + escape_dist * math.cos(best_escape_yaw)
        rec_y = our_y + escape_dist * math.sin(best_escape_yaw)

        if should_log:
            self.get_logger().warn(f"[CHASER STUCK RECOVERY] Safe path locked at {math.degrees(best_escape_yaw):.1f}°. Targeting {escape_dist:.2f}m.")

        return rec_x, rec_y, best_escape_yaw

    def truncate_to_line_of_sight(self, p_x: float, p_y: float, target_x: float, target_y: float):
        dist = math.hypot(target_x - p_x, target_y - p_y)
        
        if dist < 0.05 or self.current_state == 'STRIKING':
            return target_x, target_y

        step_size = 0.05
        steps = max(2, int(dist / step_size))
        last_safe_x, last_safe_y = p_x, p_y
        
        dx = (target_x - p_x) / steps
        dy = (target_y - p_y) / steps

        for i in range(1, steps + 1):
            tx = p_x + dx * i
            ty = p_y + dy * i
            if self.get_cell_cost(tx, ty) > MAX_SAFE_COST:
                return last_safe_x, last_safe_y
            last_safe_x, last_safe_y = tx, ty
            
        return target_x, target_y

    def find_arc_path_to_target(self, target_x: float, target_y: float, p_x: float, p_y: float, our_yaw: float):
        if self.current_state == 'STRIKING':
            return target_x, target_y

        # 1. Try the direct straight-line path first
        trunc_x, trunc_y = self.truncate_to_line_of_sight(p_x, p_y, target_x, target_y)
        if ((trunc_x - target_x)**2 + (trunc_y - target_y)**2) < 0.0025:
            return target_x, target_y

        # 2. Direct path blocked. Use vectorized candidate generation around the target.
        txs = target_x + self.arc_dx
        tys = target_y + self.arc_dy

        # 3. Calculate angles from robot to all candidates simultaneously
        angles_from_robot = np.arctan2(tys - p_y, txs - p_x)
        
        # 4. Prune backward-facing paths instantly (must stay offensive)
        turn_diffs = np.abs((angles_from_robot - our_yaw + np.pi) % (2 * np.pi) - np.pi)
        valid_mask = turn_diffs <= 1.745 # 100 degrees
        
        if not np.any(valid_mask):
            return trunc_x, trunc_y # Fallback to truncated straight line if all arcs are behind us
            
        v_txs = txs[valid_mask]
        v_tys = tys[valid_mask]
        v_r = self.arc_r[valid_mask]

        # 5. Score valid candidates using fast NumPy arrays
        dist_to_robot = np.hypot(v_txs - p_x, v_tys - p_y)
        scores = v_r + (dist_to_robot * 0.2)
        
        # Sort indices by best score (lowest is best)
        sorted_indices = np.argsort(scores)
        
        # 6. Raycast only the best paths. Stop immediately once a clear path is found!
        for idx in sorted_indices:
            cand_x = float(v_txs[idx])
            cand_y = float(v_tys[idx])
            
            arc_trunc_x, arc_trunc_y = self.truncate_to_line_of_sight(p_x, p_y, cand_x, cand_y)
            
            # If the raycast reaches the candidate point without hitting a wall, take it and exit!
            if ((arc_trunc_x - cand_x)**2 + (arc_trunc_y - cand_y)**2) < 0.0025:
                return cand_x, cand_y
                
        # If all arcs are physically blocked by the costmap, return the safest straight-line truncation
        return trunc_x, trunc_y

    def update_enemy_yaw_rate(self, enemy_yaw: float, current_time: float):
        if self.last_enemy_yaw is not None and self.last_enemy_time is not None:
            dt = current_time - self.last_enemy_time
            if 0.01 < dt < 1.0:
                raw_rate = abs(math.atan2(math.sin(enemy_yaw - self.last_enemy_yaw), math.cos(enemy_yaw - self.last_enemy_yaw))) / dt
                self.enemy_yaw_rate = 0.6 * self.enemy_yaw_rate + 0.4 * raw_rate
        self.last_enemy_yaw = enemy_yaw
        self.last_enemy_time = current_time

    def control_loop(self):
        offensive_states = ['CHASING', 'STRIKING', 'COOLDOWN']
        now_sec = self.get_clock().now().nanoseconds / 1e9
        
        if self.current_state not in offensive_states:
            if not self.was_sleeping:
                self.get_logger().info("[CHASER CALCULATOR] Yielding to Evasion Node.")
                self.was_sleeping = True
                
            self.filtered_target_x = None
            self.filtered_target_y = None
            self.flank_active_logged = False
            self.dive_active_logged = False
            self.is_recovering = False
            return

        if self.was_sleeping:
            self.get_logger().info("[CHASER CALCULATOR] Resuming Offensive Pursuit.")
            self.was_sleeping = False

        try:
            t_our = self.tf_buffer.lookup_transform(self.map_frame, self.pursuer_frame, rclpy.time.Time())
            t_carrot = self.tf_buffer.lookup_transform(self.map_frame, self.carrot_frame, rclpy.time.Time())
            t_enemy = self.tf_buffer.lookup_transform(self.map_frame, self.enemy_frame, rclpy.time.Time())
        except tf2_ros.TransformException as ex:
            return

        p_x, p_y = t_our.transform.translation.x, t_our.transform.translation.y
        our_yaw = self.get_yaw_from_quaternion(t_our.transform.rotation)
        
        c_x, c_y = t_carrot.transform.translation.x, t_carrot.transform.translation.y
        c_rot = t_carrot.transform.rotation

        e_x, e_y = t_enemy.transform.translation.x, t_enemy.transform.translation.y
        enemy_yaw = self.get_yaw_from_quaternion(t_enemy.transform.rotation)

        self.update_enemy_yaw_rate(enemy_yaw, now_sec)

        if self.filtered_target_x is None:
            self.filtered_target_x, self.filtered_target_y = c_x, c_y
        else:
            # Alpha value works as smoother when steep angle chosen
            alpha = 0.4 if self.enemy_yaw_rate > MAX_YAW_RATE_SMOOTHING else 0.8  
            self.filtered_target_x = alpha * c_x + (1.0 - alpha) * self.filtered_target_x
            self.filtered_target_y = alpha * c_y + (1.0 - alpha) * self.filtered_target_y

        angle_to_pursuer_from_enemy = math.atan2(p_y - e_y, p_x - e_x)
        rel_angle_from_front = math.atan2(math.sin(angle_to_pursuer_from_enemy - enemy_yaw), math.cos(angle_to_pursuer_from_enemy - enemy_yaw))
        abs_front_diff = abs(rel_angle_from_front)

        tactical_x, tactical_y = 0.0, 0.0

        # This here is to avoid robot even when in chasing
        if abs_front_diff < 2.094:
            if not self.flank_active_logged:
                self.get_logger().warn(f"[CHASER CALCULATOR] Orbiting {ORBITAL_FLANK_DIST}m around to the rear pocket!")
                self.flank_active_logged = True
            self.dive_active_logged = False
            
            # --- FLANK COMMITMENT LOCK ---
            # Prevent the target from teleporting left/right if the enemy wiggles near 0.0 rad
            if (now_sec - self.flank_lock_time) > 1.0:
                self.flank_direction_bias = 1.0 if rel_angle_from_front >= 0.0 else -1.0
                self.flank_lock_time = now_sec
                
            orbit_angle = angle_to_pursuer_from_enemy + (self.flank_direction_bias * 1.047)
            
            tactical_x = e_x + ORBITAL_FLANK_DIST * math.cos(orbit_angle)
            tactical_y = e_y + ORBITAL_FLANK_DIST * math.sin(orbit_angle)
            
        else:
            self.flank_active_logged = False
            
            if self.current_state == 'STRIKING':
                if not self.dive_active_logged:
                    self.get_logger().info(f"[CHASER CALCULATOR] STRIKING! Precision deep target set to {STRIKE_TARGET_DIST}m.")
                    self.dive_active_logged = True
                
                approach_angle = math.atan2(e_y - p_y, e_x - p_x)
                tactical_x = e_x - (STRIKE_TARGET_DIST * math.cos(approach_angle))
                tactical_y = e_y - (STRIKE_TARGET_DIST * math.sin(approach_angle))
                
            else:
                self.dive_active_logged = False
                tactical_x = self.filtered_target_x
                tactical_y = self.filtered_target_y

        target_x, target_y = 0.0, 0.0

        if self.current_state == 'STRIKING':
            target_x = tactical_x
            target_y = tactical_y
        else:
            dist_to_tactical = math.hypot(tactical_x - p_x, tactical_y - p_y)
            angle_to_tactical = math.atan2(tactical_y - p_y, tactical_x - p_x)
            
            if dist_to_tactical > CHASE_LEASH_DIST:
                target_x = p_x + (CHASE_LEASH_DIST * math.cos(angle_to_tactical))
                target_y = p_y + (CHASE_LEASH_DIST * math.sin(angle_to_tactical))
            else:
                target_x = tactical_x
                target_y = tactical_y

        dx = target_x - p_x
        dy = target_y - p_y
        dist_to_enemy = math.hypot(e_x - p_x, e_y - p_y)

        if math.hypot(dx, dy) > MIN_YAW_UPDATE_DIST:
            self.last_valid_yaw = math.atan2(dy, dx)

        if dist_to_enemy <= CAPTURE_ORIENT_DIST:
            rot_z = c_rot.z
            rot_w = c_rot.w
        else:
            rot_z = math.sin(self.last_valid_yaw / 2.0)
            rot_w = math.cos(self.last_valid_yaw / 2.0)

        # Finding a normal to move to 
        safe_x, safe_y = self.find_arc_path_to_target(target_x, target_y, p_x, p_y, our_yaw)


        # --- DWB-AWARE STUCK DETECTION ---
        # MAGICAL NUMBERS PRESENT HERE 
        current_cost = self.get_cell_cost(p_x, p_y)
        
        # [OPTIMIZED]: Retracted bumper projection to 0.15m to allow DWB to corner-cut smoothly
        front_x = p_x + 0.15 * math.cos(our_yaw)
        front_y = p_y + 0.15 * math.sin(our_yaw)
        front_cost = self.get_cell_cost(front_x, front_y)

        dist_to_safe_sq = (safe_x - p_x)**2 + (safe_y - p_y)**2
        dist_to_enemy_sq = (e_x - p_x)**2 + (e_y - p_y)**2
        
        target_collapsed = (dist_to_safe_sq < 0.0625) and (dist_to_enemy_sq > 0.36)
        
        # 1. Check if we are in lethal space or just corner-cutting
        is_lethal = (current_cost >= MAX_EMERGENCY_COST) or (front_cost >= MAX_EMERGENCY_COST)
        in_high_cost = (current_cost >= MAX_SAFE_COST) or (front_cost >= MAX_SAFE_COST)

        # 2. Update the Corner-Cutting Timer
        if in_high_cost:
            if self.high_cost_start_time is None:
                self.high_cost_start_time = now_sec
        else:
            self.high_cost_start_time = None

        time_in_high_cost = (now_sec - self.high_cost_start_time) if self.high_cost_start_time else 0.0

        # 3. Trigger Logic
        if not self.is_recovering:
            # Trigger instantly on Lethal/Collapsed, OR wait 0.4s for DWB to clear the corner
            if is_lethal or target_collapsed or time_in_high_cost > 0.40:
                self.is_recovering = True
        else:
            # Release recovery once we are in completely clear space
            if current_cost < (MAX_SAFE_COST - 10) and front_cost < (MAX_SAFE_COST - 10) and not target_collapsed:
                self.is_recovering = False
                self.high_cost_start_time = None

        is_stuck = self.is_recovering and (self.current_state == 'CHASING')
        should_log = (now_sec - self.last_log_time > LOG_THROTTLE_SEC)

        if is_stuck:
            if should_log:
                reason = "Lethal Crash" if is_lethal else ("Timeout in High Cost" if in_high_cost else "Path Collapsed")
                self.get_logger().warn(f"[CHASER TRIGGER] Stuck detected via {reason}. Firing Recovery!")
                
            safe_x, safe_y, safe_yaw = self.get_chaser_stuck_recovery_target(
                p_x, p_y, our_yaw, e_x, e_y, c_x, c_y, current_cost, should_log
            )
            rot_z = math.sin(safe_yaw / 2.0)
            rot_w = math.cos(safe_yaw / 2.0)
            
            if should_log:
                self.last_log_time = now_sec

            self.filtered_target_x = None
            self.filtered_target_y = None

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.broadcast_frame
        t.transform.translation.x = safe_x
        t.transform.translation.y = safe_y
        t.transform.translation.z = 0.0
        t.transform.rotation.z = rot_z
        t.transform.rotation.w = rot_w

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ChaserCalculator())
    rclpy.shutdown()

if __name__ == '__main__':
    main()