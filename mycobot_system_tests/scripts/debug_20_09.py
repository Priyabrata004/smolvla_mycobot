#!/usr/bin/env python3
import os, time, numpy as np, torch, cv2, math
from PIL import Image
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge

# SmolVLA imports (keep same)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.envs.utils import preprocess_observation

# --- Define joint limits for your robot (radians) ---
JOINT_LIMITS = [
    (-2.9, 2.9),   # link1_to_link2
    (-2.0, 2.0),   # link2_to_link3
    (-2.9, 2.9),   # link3_to_link4
    (-2.0, 2.0),   # link4_to_link5
    (-2.9, 2.9),   # link5_to_link6
    (-2.0, 2.0),   # link6_to_link6_flange
]
JOINT_LOWS = np.array([l for l, h in JOINT_LIMITS], dtype=np.float32)
JOINT_HIGHS = np.array([h for l, h in JOINT_LIMITS], dtype=np.float32)

def clamp_joints(joints):
    return [float(np.clip(j, low, high)) for j, (low, high) in zip(joints, JOINT_LIMITS)]


class SmolVLARealtime(Node):
    def __init__(self):
        super().__init__('smolvla_realtime_robot')

        # Camera subscriber
        self.subscription = self.create_subscription(
            RosImage,
            '/camera_head/color/image_raw',
            self.image_callback,
            10)
        self.bridge = CvBridge()

        # Joint trajectory publisher (adapt if your controller expects a different topic)
        self.joint_pub = self.create_publisher(
            JointTrajectory,
            '/arm_controller/joint_trajectory',
            10
        )

        # Joint state subscriber
        self.joint_names = [
            "link1_to_link2",
            "link2_to_link3",
            "link3_to_link4",
            "link4_to_link5",
            "link5_to_link6",
            "link6_to_link6_flange"
        ]
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.current_joint_positions = np.zeros(6, dtype=np.float32)
        self.prev_command = None
        self.frame_idx = 0

        # Load SmolVLA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        self.ds_meta = LeRobotDatasetMetadata("lerobot/svla_so101_pickplace")
        self.policy = SmolVLAPolicy.from_pretrained(
            "lerobot/smolvla_base", dataset_stats=self.ds_meta.stats
        )
        self.policy.to(self.device)
        self.policy.eval()
        self.get_logger().info("SmolVLA model loaded.")

        # Extract action mean/std if available (robust)
        self.action_mean = None
        self.action_std = None
        a_stats = self.ds_meta.stats.get("action", None) if hasattr(self.ds_meta, "stats") else None
        if a_stats is not None:
            # some metadata stores lists/tensors; convert to numpy arrays when present
            mean = a_stats.get("mean", None)
            std  = a_stats.get("std", None)
            if mean is not None and std is not None:
                import numpy as _np
                self.action_mean = _np.asarray(mean, dtype=_np.float32)
                self.action_std  = _np.asarray(std,  dtype=_np.float32)
                self.get_logger().info(f"Loaded action stats: mean.shape={self.action_mean.shape}, std.shape={self.action_std.shape}")
            else:
                self.get_logger().info("Action stats present but missing mean/std; will use raw outputs.")
        else:
            self.get_logger().info("No action stats found in dataset metadata; will use raw outputs.")

    def joint_state_callback(self, msg: JointState):
        """Updates the current joint positions from /joint_states."""
        try:
            # map joint names robustly
            indices = [msg.name.index(j) for j in self.joint_names]
            self.current_joint_positions = np.array([msg.position[i] for i in indices], dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f"Joint state callback error: {e}")

    def image_callback(self, msg: RosImage):
        self.frame_idx += 1

        # Convert ROS image to RGB numpy array
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image: {e}")
            return

        # Quick sanity: image stats
        img_resized = cv2.resize(cv_image, (256, 256))
        self.get_logger().info(f"Frame {self.frame_idx}: image shape={img_resized.shape}, dtype={img_resized.dtype}, min={img_resized.min()}, max={img_resized.max()}")

        # Optional: save a sample image occasionally for offline debug
        if self.frame_idx % 150 == 0:
            os.makedirs('/tmp/smolvla_debug', exist_ok=True)
            sample_path = f'/tmp/smolvla_debug/frame_{self.frame_idx}.png'
            # convert RGB->BGR to save consistently with OpenCV tools
            cv2.imwrite(sample_path, cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))
            self.get_logger().info(f"Saved debug image to {sample_path}")

        # Basic red-block detector (sanity check that camera can see the red block)
        try:
            hsv = cv2.cvtColor(img_resized, cv2.COLOR_RGB2HSV)
            lower1 = np.array([0, 60, 50]); upper1 = np.array([10, 255, 255])
            lower2 = np.array([170, 60, 50]); upper2 = np.array([180, 255, 255])
            mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)
            red_fraction = float(mask.sum()) / (mask.size * 255.0)
            self.get_logger().info(f"Red fraction in image: {red_fraction:.5f}")
        except Exception as e:
            self.get_logger().warn(f"Red detector failed: {e}")
            red_fraction = 0.0

        # Dummy state (zeros if no proprioception in dataset)
        state_dim = int(self.policy.normalize_inputs.buffer_observation_state.mean.shape[-1])
        state = np.zeros((state_dim,), dtype=np.float32)

        # Build observation — make task lowercase, as some preprocessors are case-sensitive
        observation = {"pixels": img_resized, "agent_pos": state}
        observation = preprocess_observation(observation)  # uses policy dataset-stat normalization
        observation = {k: v.to(self.device) for k, v in observation.items()}

        # Provide task in multiple acceptable forms (some models accept list[str], others 'task' tokenized via preprocess)
        # Try a simple, consistent form:
        observation["task"] = ["move upwards"]

        # Run inference
        with torch.inference_mode():
            t0 = time.time()
            action = self.policy.select_action(observation)
            dt = time.time() - t0

        self.get_logger().info(f"Inference time: {dt*1000:.1f} ms, action shape: {tuple(action.shape)}")

        # Convert action → safe joint targets
        joint_positions = self.interpret_action(action)

        if joint_positions is not None:
            # Publish a 2-point trajectory: current -> target for smooth interpolation
            traj_msg = JointTrajectory()
            traj_msg.joint_names = self.joint_names

            p0 = JointTrajectoryPoint()
            p0.positions = list(self.current_joint_positions.tolist())
            p0.time_from_start.sec = 0

            p1 = JointTrajectoryPoint()
            p1.positions = joint_positions
            p1.time_from_start.sec = 1  # 1 second to reach target (adjust if too slow/fast)

            traj_msg.points = [p0, p1]
            self.joint_pub.publish(traj_msg)
            self.get_logger().info(f"Published trajectory: start={p0.positions}, target={p1.positions}")

    '''def interpret_action(self, action_tensor):
        """Robust conversion of model action -> joint positions (radians), with detection of semantics."""
        try:
            action = action_tensor[0].cpu().numpy().astype(np.float32)  # shape (6,)
            self.get_logger().info(f"Raw action values: {action}")

            # Sanity check NaNs / infs
            if not np.isfinite(action).all():
                self.get_logger().warn("Action contains NaN/Inf. Ignoring.")
                return None

            # Heuristic 1: if action already lies inside joint limits -> probably ABSOLUTE joint positions (radians)
            if np.all(action >= JOINT_LOWS - 1e-3) and np.all(action <= JOINT_HIGHS + 1e-3):
                self.get_logger().info("Interpreting action as ABSOLUTE joint positions (radians).")
                next_state = action
            else:
                # Otherwise treat as deltas or degrees or normalized values.
                max_abs = float(np.max(np.abs(action)))
                self.get_logger().info(f"Max abs(action) = {max_abs:.3f}")

                if max_abs > 10.0:
                    # values look large -> likely degrees. Convert to radians and scale down.
                    self.get_logger().info("Large action magnitudes -> converting degrees->radians and scaling.")
                    action_rad = np.radians(action)
                    delta = 0.05 * action_rad
                elif max_abs > 2.0:
                    # moderate values -> probably rad deltas
                    self.get_logger().info("Moderate magnitudes -> treating as rad deltas (scaling by 0.2).")
                    delta = 0.2 * action
                else:
                    # small values -> likely normalized in [-1,1] or small rad -> scale to safe delta
                    self.get_logger().info("Small magnitudes -> treating as normalized/small rad; scaling by 0.15.")
                    delta = 0.15 * action

                next_state = self.current_joint_positions + delta

            # Clamp to joint limits
            next_state = np.clip(next_state, JOINT_LOWS, JOINT_HIGHS)

            # Smooth (low-pass) to reduce jitter
            if self.prev_command is None:
                self.prev_command = self.current_joint_positions.copy()
            alpha = 0.5  # 0..1 smaller -> stronger smoothing
            smoothed = alpha * next_state + (1.0 - alpha) * self.prev_command
            self.prev_command = smoothed.copy()

            # Final clamp & return Python list
            final = np.clip(smoothed, JOINT_LOWS, JOINT_HIGHS).tolist()
            self.get_logger().info(f"Interpreted next_state (smoothed/clamped): {final}")
            return final
        except Exception as e:
            self.get_logger().warn(f"Failed to interpret action: {e}")
            return None'''
    def interpret_action(self, action_tensor):
        """Convert SmolVLA action output -> joint positions (radians)."""
   
        try:
        # 1) Get raw normalized action (from model output)
            raw = action_tensor[0].cpu().numpy().astype(np.float32)
            self.get_logger().info(f"Raw action values: {raw}")

            # 2) Denormalize using dataset stats
            #unnorm = raw * self.action_std + self.action_mean
            #unnorm = raw
            #self.get_logger().info(f"Denormalized action (absolute radians): {unnorm}")
            max_abs = np.max(np.abs(raw))
            if max_abs > 20:
                self.get_logger().info("Actions look like DEGREES, converting to radians.")
                unnorm = np.radians(raw)
            elif max_abs > 3:
                self.get_logger().info("Actions look like RAD deltas, scaling x0.1.")
                unnorm = 0.1 * raw
            else:
                self.get_logger().info("Actions look like RADIANS already.")
                unnorm = raw
            self.get_logger().info(f"After converting: {unnorm}")
            # 3) Ensure correct dimension
            n_joints = len(self.joint_names)
            if unnorm.shape[0] != n_joints:
                self.get_logger().warn(
                    f"Action dim {unnorm.shape[0]} != #joints {n_joints}, adapting by slice/pad."
                )
                if unnorm.shape[0] > n_joints:
                    unnorm = unnorm[:n_joints]
                else:
                    unnorm = np.pad(unnorm, (0, n_joints - unnorm.shape[0]), "constant")

            # 4) Clamp to safe joint limits
            next_state = np.clip(unnorm, JOINT_LOWS, JOINT_HIGHS)

            # 5) Smooth with exponential moving average
            if self.prev_command is None:
                self.prev_command = self.current_joint_positions.copy()
            alpha = 0.5  # smoothing factor
            smoothed = alpha * next_state + (1.0 - alpha) * self.prev_command
            self.prev_command = smoothed.copy()

            final = np.clip(smoothed, JOINT_LOWS, JOINT_HIGHS).tolist()
            self.get_logger().info(f"Final joint targets: {final}")

            return final

        except Exception as e:
            self.get_logger().warn(f"Failed to interpret action: {e}")
            return None



def main(args=None):
    rclpy.init(args=args)
    node = SmolVLARealtime()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
