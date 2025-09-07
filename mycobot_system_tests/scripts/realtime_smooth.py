#!/usr/bin/env python3
import os, time, numpy as np, torch, cv2, math
from PIL import Image
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge

# SmolVLA imports
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.envs.utils import preprocess_observation


# --- Define joint limits for your robot (adapt if needed) ---
JOINT_LIMITS = [
    (-2.9, 2.9),   # link1_to_link2
    (-2.0, 2.0),   # link2_to_link3
    (-2.9, 2.9),   # link3_to_link4
    (-2.0, 2.0),   # link4_to_link5
    (-2.9, 2.9),   # link5_to_link6
    (-2.0, 2.0),   # link6_to_link6_flange
]

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

        # Joint trajectory publisher
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

        self.current_joint_positions = np.zeros(6)

        # Load SmolVLA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        ds_meta = LeRobotDatasetMetadata("lerobot/svla_so101_pickplace")
        self.policy = SmolVLAPolicy.from_pretrained(
            "lerobot/smolvla_base", dataset_stats=ds_meta.stats
        )
        self.policy.to(self.device)
        self.policy.eval()
        self.get_logger().info("SmolVLA model loaded.")


    def joint_state_callback(self, msg: JointState):
        """Updates the current joint positions from /joint_states."""
        try:
            indices = [msg.name.index(j) for j in self.joint_names]
            self.current_joint_positions = np.array([msg.position[i] for i in indices])
        except Exception as e:
            self.get_logger().warn(f"Joint state callback error: {e}")


    def image_callback(self, msg: RosImage):
        # Convert ROS image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        img_resized = cv2.resize(cv_image, (256, 256))

        # Dummy state (zeros if no proprioception in dataset)
        state_dim = self.policy.normalize_inputs.buffer_observation_state.mean.shape[-1]
        state = np.zeros((state_dim,), dtype=np.float32)

        # Build observation
        observation = {"pixels": img_resized, "agent_pos": state}
        observation = preprocess_observation(observation)
        observation = {k: v.to(self.device) for k, v in observation.items()}
        observation["task"] = ["Pick up the red block"]

        # Run inference
        with torch.inference_mode():
            t0 = time.time()
            action = self.policy.select_action(observation)
            dt = time.time() - t0

        self.get_logger().info(
            f"Inference time: {dt*1000:.1f} ms, action shape: {tuple(action.shape)}"
        )

        # Convert action → smooth trajectory
        joint_positions = self.interpret_action(action)
        if joint_positions is not None:
            traj_msg = JointTrajectory()
            traj_msg.joint_names = self.joint_names

            point = JointTrajectoryPoint()
            point.positions = joint_positions
            point.time_from_start.sec = 2   # smoother 2s execution

            traj_msg.points.append(point)
            self.joint_pub.publish(traj_msg)
            self.get_logger().info(f"Published next joint state: {joint_positions}")


    def interpret_action(self, action_tensor):
        """Convert SmolVLA action to safe joint targets (delta-based)."""
        try:
            action = action_tensor[0].cpu().numpy()

            action_rad = np.radians(action)

            # Scale deltas to avoid jumps
            delta = 0.1 * action_rad

            # Apply delta to current state
            next_state = self.current_joint_positions + delta

            # Clamp inside joint limits
            next_state = clamp_joints(next_state)

            return next_state
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
