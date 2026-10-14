#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import OccupancyGrid
import tf2_ros
from geometry_msgs.msg import TransformStamped

class ChaserCalculator(Node):
    """
    ChaserCalculator dynamically generates interception paths ("carrot points") for the 
    robot to follow when in the CHASING state. It intercepts the enemy's projected path,
    smooths target teleportation on sharp turns, and features an Anti-Ramming Flank Route 
    to dodge head-on collisions when the enemy suddenly spins to face the chaser[cite: 6].
    """
    def __init__(self):
        super().__init__('chaser_calculator')

        # --- Dynamic Namespace Configuration ---
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

        # The 'carrot_frame' is the raw target point projected behind/intercepting the enemy[cite: 6]
        carrot_p = self.get_parameter('carrot_frame').value
        self.carrot_frame = carrot_p if carrot_p else (f'{self.enemy_ns}/target_point' if self.enemy_ns else 'target_point')

        # The 'broadcast_frame' is the final, filtered Nav2 goal sent to goal_sender[cite: 6]
        self.broadcast_frame = self.get_parameter('broadcast_frame').value

        # --- Tuning Parameters ---
        # limits how far ahead the intermediate goal is placed to prevent Nav2 from taking wild detours[cite: 6]
        self.short_goal_dist = 0.8
        # Distance at which the robot stops aiming 'at' the target and instead aligns its orientation with it[cite: 6]
        self.capture_orientation_dist = 0.35

        # --- State & Map Variables ---
        self.current_state = 'CHASING'
        self.costmap = None
        self.last_valid_yaw = 0.0

        # --- ENHANCEMENT: Enemy Spin Tracking ---
        # Tracks how fast the enemy is turning to apply dynamic smoothing[cite: 6]
        self.last_enemy_yaw = None
        self.last_enemy_time = None
        self.enemy_yaw_rate = 0.0

        # --- ENHANCEMENT: Target TF Smoothing ---
        # Prevents the chase target from instantly teleporting across the room if the enemy spins 180 degrees[cite: 6]
        self.filtered_target_x = None
        self.filtered_target_y = None

        # --- TF2 & Subscriptions ---
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        state_topic = f'/{self.robot_ns}/robot_state' if self.robot_ns else '/robot_state'
        costmap_topic = f'/{self.robot_ns}/global_costmap/costmap' if self.robot_ns else '/global_costmap/costmap'

        self.create_subscription(String, state_topic, self.state_cb, 10)
        self.create_subscription(OccupancyGrid, costmap_topic, self.costmap_cb, 10)

        # 10Hz Control Loop[cite: 6]
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Smart Chaser Calculator initialized (Flank Routing & Spin Guard Active).")

    def state_cb(self, msg):
        """Updates the internal state machine flag ('CHASING' vs 'EVADING')[cite: 6]"""
        self.current_state = msg.data

    def costmap_cb(self, msg: OccupancyGrid):
        """Caches the latest Nav2 Global Costmap for collision checking[cite: 6]"""
        self.costmap = msg

    def get_yaw_from_quaternion(self, q):
        """Converts a geometry_msgs Quaternion into a planar yaw angle (radians)[cite: 6]"""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def is_pose_safe(self, x: float, y: float, max_cost: int = 50) -> bool:
        """
        Checks if a specific global coordinate is safe to travel to based on the Nav2 costmap.
        Also enforces a hard 0.75m boundary margin so the robot doesn't target map edges.[cite: 6]
        """
        if self.costmap is None: return True
        info = self.costmap.info
        gx = int((x - info.origin.position.x) / info.resolution)
        gy = int((y - info.origin.position.y) / info.resolution)

        # MAP BOUNDARY GUARD: Treat the outer 15 cells as lethal walls[cite: 6]
        MARGIN = 15
        if gx < MARGIN or gx >= (info.width - MARGIN) or \
           gy < MARGIN or gy >= (info.height - MARGIN):
            return False

        cost = self.costmap.data[gy * info.width + gx]
        return 0 <= cost < max_cost

    def project_to_free_space(self, start_x: float, start_y: float, target_x: float, target_y: float):
        """
        If the calculated chase point accidentally falls inside a wall, this raycasts backwards
        from the wall toward the robot's current position to find the nearest safe open space.[cite: 6]
        """
        if self.is_pose_safe(target_x, target_y, max_cost=50):
            return target_x, target_y

        dist = math.hypot(target_x - start_x, target_y - start_y)
        if dist < 0.05: return start_x, start_y

        # Raycast backwards in 5% increments[cite: 6]
        for i in range(1, 20):
            alpha = 1.0 - (i / 20.0)
            px = start_x + alpha * (target_x - start_x)
            py = start_y + alpha * (target_y - start_y)
            if self.is_pose_safe(px, py, max_cost=50):
                return px, py

        return start_x, start_y

    def update_enemy_yaw_rate(self, enemy_yaw: float, current_time: float):
        """
        Calculates the angular velocity of the enemy. Used to trigger the 
        filter smoothing if the enemy begins spinning wildly.[cite: 6]
        """
        if self.last_enemy_yaw is not None and self.last_enemy_time is not None:
            dt = current_time - self.last_enemy_time
            if 0.01 < dt < 1.0:
                raw_rate = abs(math.atan2(math.sin(enemy_yaw - self.last_enemy_yaw), math.cos(enemy_yaw - self.last_enemy_yaw))) / dt
                self.enemy_yaw_rate = 0.6 * self.enemy_yaw_rate + 0.4 * raw_rate
        self.last_enemy_yaw = enemy_yaw
        self.last_enemy_time = current_time

    def control_loop(self):
        """
        Main calculation loop. Projects the Nav2 goal toward the enemy's rear target point,
        filters jerky movements, and routes around the enemy's front cone to prevent head-on crashes.[cite: 6]
        """
        # Standby if not actively chasing[cite: 6]
        if self.current_state != 'CHASING':
            self.filtered_target_x = None
            self.filtered_target_y = None
            return

        # Fetch required transforms[cite: 6]
        try:
            t_our = self.tf_buffer.lookup_transform(self.map_frame, self.pursuer_frame, rclpy.time.Time())
            t_carrot = self.tf_buffer.lookup_transform(self.map_frame, self.carrot_frame, rclpy.time.Time())
            t_enemy = self.tf_buffer.lookup_transform(self.map_frame, self.enemy_frame, rclpy.time.Time())
        except tf2_ros.TransformException:
            return

        p_x, p_y = t_our.transform.translation.x, t_our.transform.translation.y
        c_x, c_y = t_carrot.transform.translation.x, t_carrot.transform.translation.y
        c_rot = t_carrot.transform.rotation

        e_x, e_y = t_enemy.transform.translation.x, t_enemy.transform.translation.y
        enemy_yaw = self.get_yaw_from_quaternion(t_enemy.transform.rotation)

        now_sec = self.get_clock().now().nanoseconds / 1e9
        self.update_enemy_yaw_rate(enemy_yaw, now_sec)

        # --- ENHANCEMENT 1: Target Smoothing ---
        # If the enemy spins 180 degrees, their raw rear "carrot_frame" instantly teleports 
        # to the other side of the map. This filter drags the target smoothly through space 
        # so Nav2 doesn't abort from impossible goal jumps.[cite: 6]
        if self.filtered_target_x is None:
            self.filtered_target_x, self.filtered_target_y = c_x, c_y
        else:
            # Drop alpha heavily (smoother movement) if the enemy is spinning fast (> 1.0 rad/s)[cite: 6]
            alpha = 0.4 if self.enemy_yaw_rate > 1.0 else 0.8  
            self.filtered_target_x = alpha * c_x + (1.0 - alpha) * self.filtered_target_x
            self.filtered_target_y = alpha * c_y + (1.0 - alpha) * self.filtered_target_y

        dx = self.filtered_target_x - p_x
        dy = self.filtered_target_y - p_y
        dist = math.hypot(dx, dy)

        # --- ENHANCEMENT 2: Anti-Ramming Flank Routing ---
        # We calculate our angle relative to the enemy's front nose.[cite: 6]
        angle_to_pursuer_from_enemy = math.atan2(p_y - e_y, p_x - e_x)
        front_cone_diff = abs(math.atan2(math.sin(angle_to_pursuer_from_enemy - enemy_yaw), math.cos(angle_to_pursuer_from_enemy - enemy_yaw)))

        # If the enemy spins to face us (< 60 deg) while we are close (< 1.8m), 
        # a direct chase path would result in a head-on collision.[cite: 6]
        if front_cone_diff < math.radians(60.0) and dist < 1.8:
            # Find which side of the enemy is slightly closer and set the target to their flank
            flank_side = (angle_to_pursuer_from_enemy - enemy_yaw) > 0
            flank_offset_angle = enemy_yaw + (math.pi / 2.0 if flank_side else -math.pi / 2.0)
            
            # Offset the goal 1.0m to their side
            target_x = e_x + 1.0 * math.cos(flank_offset_angle)
            target_y = e_y + 1.0 * math.sin(flank_offset_angle)
        else:
            # Standard Linear Lead Pursuit
            # Shorten the goal to prevent Nav2 from taking wide detours[cite: 6]
            lead_dist = min(self.short_goal_dist, max(0.4, dist * 0.7))
            if dist > lead_dist:
                ratio = lead_dist / dist
                target_x = p_x + dx * ratio
                target_y = p_y + dy * ratio
            else:
                target_x = self.filtered_target_x
                target_y = self.filtered_target_y

        # Maintain heading logic: Stops the robot from wildly spinning 180 degrees 
        # when it overshoots the target by a few millimeters[cite: 6]
        if dist > 0.15:
            self.last_valid_yaw = math.atan2(dy, dx)

        # Terminal Alignment: Once extremely close (< 0.35m), stop pointing at the goal 
        # and match the enemy's orientation to prep for capture/docking[cite: 6]
        if dist <= self.capture_orientation_dist:
            rot_z = c_rot.z
            rot_w = c_rot.w
        else:
            rot_z = math.sin(self.last_valid_yaw / 2.0)
            rot_w = math.cos(self.last_valid_yaw / 2.0)

        # Final pass: Ensure the calculated goal isn't inside a solid wall[cite: 6]
        safe_x, safe_y = self.project_to_free_space(p_x, p_y, target_x, target_y)

        # Broadcast the final processed goal to the goal_sender[cite: 6]
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.broadcast_frame
        t.transform.translation.x = safe_x
        t.transform.translation.y = safe_y
        t.transform.rotation.z = rot_z
        t.transform.rotation.w = rot_w

        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(ChaserCalculator())
    rclpy.shutdown()

if __name__ == '__main__':
    main()