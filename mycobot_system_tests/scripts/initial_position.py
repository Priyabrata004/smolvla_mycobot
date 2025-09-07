'''import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

class InitPose(Node):
    def __init__(self):
        super().__init__('init_pose_sender')
        self.pub = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.timer = self.create_timer(1.0, self.send_pose)

    def send_pose(self):
        traj = JointTrajectory()
        traj.joint_names = [
            "joint_1", "joint_2", "joint_3",
            "joint_4", "joint_5", "joint_6"
        ]

        # A rough "look down" pose — adjust if needed
        point = JointTrajectoryPoint()
        point.positions = [0.0, 0.8, -1.2, 1.5, -1.5, 0.0]
        point.time_from_start.sec = 3
        traj.points.append(point)

        self.pub.publish(traj)
        self.get_logger().info("Sent init pose")
        self.timer.cancel()

rclpy.init()
node = InitPose()
rclpy.spin(node)
node.destroy_node()
rclpy.shutdown()'''

#!/usr/bin/env python3

import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration


class LookDownInitializer(Node):
    def __init__(self):
        super().__init__('look_down_initializer')

        self.arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory'
        )

        self.gripper_client = ActionClient(
            self,
            GripperCommand,
            '/gripper_action_controller/gripper_cmd'
        )

        self.joint_names = [
            'link1_to_link2', 'link2_to_link3', 'link3_to_link4',
            'link4_to_link5', 'link5_to_link6', 'link6_to_link6_flange'
        ]

        # Replace with your actual look-down pose
        self.look_down_pose = [0.0, 0.0, 0.0, -0.5, 0.0, 0.0]

        # Wait for servers and then send command
        self.wait_for_servers()

    def wait_for_servers(self):
        self.get_logger().info('Waiting for action servers...')
        self.arm_client.wait_for_server()
        self.gripper_client.wait_for_server()
        self.get_logger().info('Action servers available! Moving arm to look-down pose...')
        self.send_look_down_command()

    def send_look_down_command(self):
        point = JointTrajectoryPoint()
        point.positions = self.look_down_pose
        point.time_from_start = Duration(sec=3)

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        goal_msg.trajectory.points = [point]

        send_goal_future = self.arm_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.on_look_down_sent)

    def on_look_down_sent(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Look-down pose goal rejected')
            rclpy.shutdown()
            return

        self.get_logger().info('Look-down goal accepted. Waiting for result...')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.on_look_down_done)

    def on_look_down_done(self, future):
        result = future.result().result
        self.get_logger().info('Look-down movement complete.')
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = LookDownInitializer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

