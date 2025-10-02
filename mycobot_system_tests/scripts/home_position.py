import time
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory, GripperCommand
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration


class InitialPositionInitializer(Node):
    def __init__(self):
        super().__init__('initial_position_initializer')

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

        # Replace with your robot's actual initial/home position
        self.initial_pose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Wait for servers and then send command
        self.wait_for_servers()

    def wait_for_servers(self):
        self.get_logger().info('Waiting for action servers...')
        self.arm_client.wait_for_server()
        self.gripper_client.wait_for_server()
        self.get_logger().info('Action servers available! Moving arm to initial pose...')
        self.send_initial_position_command()

    def send_initial_position_command(self):
        point = JointTrajectoryPoint()
        point.positions = self.initial_pose
        point.time_from_start = Duration(sec=3)

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        goal_msg.trajectory.points = [point]

        send_goal_future = self.arm_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.on_initial_position_sent)

    def on_initial_position_sent(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Initial pose goal rejected')
            rclpy.shutdown()
            return
        self.get_logger().info('Initial pose goal accepted!')


def main(args=None):
    rclpy.init(args=args)
    node = InitialPositionInitializer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
