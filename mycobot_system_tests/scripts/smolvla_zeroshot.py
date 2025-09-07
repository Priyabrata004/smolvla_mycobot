#!/usr/bin/env python3
import os, time, numpy as np, torch, cv2
from PIL import Image
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge

# SmolVLA imports
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.envs.utils import preprocess_observation

class SmolVLARealtime(Node):
    def __init__(self):
        super().__init__('smolvla_realtime_robot')
        self.subscription = self.create_subscription(
            RosImage,
            '/camera_head/color/image_raw'  # adjust to your camera topic
            self.image_callback,
            10)
        self.bridge = CvBridge()

        # Robot publisher
        self.joint_pub = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6']  # adjust for MyCobot

        # Load SmolVLA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using device: {self.device}")

        ds_meta = LeRobotDatasetMetadata("lerobot/svla_so101_pickplace")
        self.policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base", dataset_stats=ds_meta.stats)
        self.policy.to(self.device)
        self.policy.eval()
        self.get_logger().info("SmolVLA model loaded.")

    def image_callback(self, msg: RosImage):
        # Convert ROS image to NumPy array
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        img_resized = cv2.resize(cv_image, (256, 256))

        # Dummy robot state (zeros if no proprioception yet)
        state_dim = self.policy.normalize_inputs.buffer_observation_state.mean.shape[-1]
        state = np.zeros((state_dim,), dtype=np.float32)

        # Build observation
        observation = {"pixels": img_resized, "agent_pos": state}
        observation = preprocess_observation(observation)
        observation = {k: v.to(self.device, non_blocking=(self.device.type=="cuda")) for k, v in observation.items()}

        # Zero-shot natural language instruction
        observation["task"] = ["Pick up the red block"]

        # Inference
        with torch.inference_mode():
            t0 = time.time()
            action = self.policy.select_action(observation)
            dt = time.time() - t0

        self.get_logger().info(f"Inference time: {dt*1000:.1f} ms, action shape: {tuple(action.shape)}")

        # Convert action to joint positions and publish
        joint_positions = self.interpret_action(action)
        if joint_positions is not None:
            traj_msg = JointTrajectory()
            traj_msg.joint_names = self.joint_names
            point = JointTrajectoryPoint()
            point.positions = joint_positions
            point.time_from_start.sec = 1
            traj_msg.points.append(point)
            self.joint_pub.publish(traj_msg)
            self.get_logger().info(f"Published joint positions: {joint_positions}")

    def interpret_action(self, action_tensor):
        """
        Convert SmolVLA action to robot joint positions.
        If action is already joint angles: return list.
        If action is end-effector delta: implement IK.
        """
        try:
            joint_positions = action_tensor[0].cpu().numpy().tolist()  # adapt if necessary
            return joint_positions
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
