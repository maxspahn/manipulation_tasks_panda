#!/usr/bin/env python3
'''
@Author: Stan Zwinkels, Xinjie (Richard) Liu
@Date: 2022-4-25 16:46:30
@LastEditTime: 2022-5-19 17:51:32
@Description: communication node of the find_object_2d and board_detector
'''

import rospy
from geometry_msgs.msg import PoseStamped, WrenchStamped, PointStamped, QuaternionStamped, Pose, TransformStamped
from std_msgs.msg import Float32MultiArray, Bool
from sensor_msgs.msg import JointState
import numpy as np
import quaternion
import time
import math
import dynamic_reconfigure.client
from pyquaternion import Quaternion
from sympy import Point, Line
from manipulation_helpers.pose_transform_functions import orientation_2_quaternion, pose_2_transformation, position_2_array, array_quat_2_pose, transformation_2_pose, transform_pose
class BoardDetector():
    '''
    a wrapper class for board detection
    '''
    def __init__(self):
        self.r = rospy.Rate(100)

        self.force_threshold = 7

        self.sub = rospy.Subscriber('/cartesian_pose', PoseStamped, self.update_pose)
        self.force_feedback_sub = rospy.Subscriber('/force_torque_ext', WrenchStamped, self.force_feedback_checker)
        self.joint_states_sub = rospy.Subscriber("/joint_states", JointState, self.joint_states_callback)
        self.transform_icp_sub = rospy.Subscriber('/trans_rot', PoseStamped, self.transform_icp_callback)

        self.goal_pub = rospy.Publisher('/equilibrium_pose', PoseStamped, queue_size=0)
        self.pose_ref_2_new_pub = rospy.Publisher('/pose_ref_2_new', PoseStamped, queue_size=0)

        self.pose_ref_2_new = PoseStamped()
        self.collided = False
        self.board_width = 0.16
        self.board_length = 0.26
        self.board_height = 0.07
        self.clearance = 0.07
        self.detected = False
        self.force_feedback = None
        self.torque_z = None
        self.curr_pos = np.empty([3])
        self.curr_ori = np.empty([3])
        self.points_x = []
        self.points_y = []
        self.trajz = []
        self.ori_x = np.quaternion(1, 0, 0, 0)
        self.ori_y = np.quaternion(1, 0, 0, 0)
        self.coll_points = [np.array([[0, 0, 0]])]
        self.coll_points_to_save = []
        self.coll_oris = [np.array([[0, 0, 0, 0]])]
        self.coll_oris_to_save = []
        self.K_pos = 600
        self.K_z = 1000
        self.K_ori = 50

        self.trans_ref_guess = [0.5, 0.1512, 0.16]
        self.rot_ref_guess = np.quaternion(np.sqrt(0.5), 0, 0, -np.sqrt(0.5)) * np.quaternion(0.9999, 0, 0, 0.0111)

        self.trans_cam = np.array([0.483, 0.021, 0.58])
        self.rot_cam = np.quaternion(0.006, 0.734, -0.679, 0.006)

        self.pose_icp = PoseStamped()
        self.pose_icp.pose.orientation.w = 1
        self.pose_box = PoseStamped()

    def set_stiffness(self, k_t1, k_t2, k_t3, k_r1, k_r2, k_r3, k_ns):

        set_K = dynamic_reconfigure.client.Client('/dynamic_reconfigure_compliance_param_node', config_callback=None)
        set_K.update_configuration({"translational_stiffness_X": k_t1})
        set_K.update_configuration({"translational_stiffness_Y": k_t2})
        set_K.update_configuration({"translational_stiffness_Z": k_t3})
        set_K.update_configuration({"rotational_stiffness_X": k_r1})
        set_K.update_configuration({"rotational_stiffness_Y": k_r2})
        set_K.update_configuration({"rotational_stiffness_Z": k_r3})
        set_K.update_configuration({"nullspace_stiffness": k_ns})

    def joint_states_callback(self, data):
        self.curr_joint = data.position[:7]

    def update_pose(self, data):
        self.curr_pos = np.array([data.pose.position.x, data.pose.position.y, data.pose.position.z])
        self.curr_ori = np.array([data.pose.orientation.w, data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z])

    def force_feedback_checker(self, feedback):
        force = feedback.wrench.force
        torque = feedback.wrench.torque
        self.torque_z = torque.z
        self.force_feedback = np.linalg.norm(np.array([force.x, force.y, force.z]))
        if self.force_feedback > self.force_threshold:
            # print("force exceeded,current point is", self.curr_pos, self.curr_ori)
            self.collided = True
            # print("current point ", self.curr_pos)
            self.coll_points.append(self.curr_pos)
            # print(self.coll_points, self.curr_pos)
            self.coll_oris.append(self.curr_ori)
            # print(self.coll_oris, self.curr_ori)
            self.collided = False
        else:
            self.collided = False

    def transform_icp_callback(self, pose_icp):
        self.pose_icp = pose_icp

    # TODO this would be much better in a config file since the points are now fixed
    # points defined with respect to local initially detected base frame
    def get_trajectories(self):
        # For x-axis approach we just want the orientation to be aligned with local frame
        self.ori_x = np.quaternion(0, 1, 0, 0)
        # For y-axis orientation just a 90 deg rotation about z-axis to touch with same flat side of flange
        # These orientations are fixed throughout approach
        self.ori_y = np.quaternion(0, np.sqrt(0.5), np.sqrt(0.5), 0)

        # I'm assuming we are putting the coordinate systems in a corner,
        # and that x-axis goes along length and y along width

        # Point outside and above box
        p1x = [-self.clearance, self.board_width/2, self.clearance]
        p2x = [p1x[0], p1x[1], -self.board_height*0.6]
        p3x = [self.board_width/2, p2x[1], p2x[2]]

        self.points_x = [p1x, p2x, p3x]
        self.poses_x = []
        self.poses_x[:] = [array_quat_2_pose(point_x, self.ori_x) for point_x in self.points_x]

        p1y = [self.board_length*0.7, -self.clearance, self.clearance]
        p2y = [p1y[0], p1y[1], -self.board_height*0.7]
        p3y = [p2y[0], self.board_width * 0.35, p2y[2]]

        self.points_y = [p1y, p2y, p3y]
        self.poses_y = []
        self.poses_y[:] = [array_quat_2_pose(point_y, self.ori_y) for point_y in self.points_y]

        return

    def execute_trajectory(self, poses):
        for pose in poses:
            if pose == poses[-1]: self.set_stiffness(self.K_pos, self.K_pos, self.K_z, self.K_ori, self.K_ori, 0.0, 0.0)
            else:      self.set_stiffness(self.K_pos, self.K_pos, self.K_z, self.K_ori, self.K_ori, self.K_ori, 0.0)

            self.go_to_pose(pose)

        self.coll_points_to_save.append(self.coll_points[-1])
        self.coll_oris_to_save.append(self.coll_oris[-1])

        self.go_to_pose(poses[0])
    
    def execute_touch(self, offline):
        self.get_trajectories()

        # 'Offline' you manually input a guess of the box pose to then save an accurate pose from the touch (to have
        # the reference pose of the recordings), 'online' the perception is used to get the 'initial guess'
        if offline:
            file_suffix = "ref"
            pose_box_ref_guess = array_quat_2_pose(self.trans_ref_guess, self.rot_ref_guess)
            transform = pose_2_transformation(pose_box_ref_guess)
        else:
            file_suffix = "new"
            pose_cam = array_quat_2_pose(self.trans_cam, self.rot_cam)
            transform_cam = pose_2_transformation(pose_cam)
            transform_icp = pose_2_transformation(self.pose_icp)
            transform_ref = np.load("transform_box_ref.npy")
            transform = transform_cam @ transform_icp @ np.linalg.inv(transform_cam) @ transform_ref

        poses_x_transformed = []
        poses_x_transformed[:] = [transform_pose(pose_x, transform) for pose_x in self.poses_x]
        self.execute_trajectory(poses_x_transformed)

        poses_y_transformed = []
        poses_y_transformed[:] = [transform_pose(pose_y, transform) for pose_y in self.poses_y]
        self.execute_trajectory(poses_y_transformed)

        np.save(f"coll_points_{file_suffix}", self.coll_points_to_save)
        np.save(f"coll_oris_{file_suffix}", self.coll_oris_to_save)
        self.pose_box = self.box_pose_from_2_points(self.coll_points_to_save[0], self.coll_oris_to_save[0],
                                                       self.coll_points_to_save[1], self.coll_oris_to_save[1])
        transform_box = pose_2_transformation(self.pose_box)
        np.save(f"transform_box_{file_suffix}", transform_box)

        return

    # basically copied from lfd, should we import or inherit instead?
    def go_to_pose(self, goal_pose):
        # the goal pose should be of type PoseStamped. E.g. goal_pose=PoseStampled()
        start = self.curr_pos
        start_ori = self.curr_ori
        goal_ = np.array([goal_pose.pose.position.x, goal_pose.pose.position.y, goal_pose.pose.position.z])
        # interpolate from start to goal with attractor distance of approx 1 cm
        print("start is here", start, "goal is here", goal_)
        squared_dist = np.sum(np.subtract(start, goal_) ** 2, axis=0)
        dist = np.sqrt(squared_dist)
        print("dist", dist)
        interp_dist = 0.001 /2  # [m]
        step_num_lin = math.floor(dist / interp_dist)

        print("num of steps linear", step_num_lin)

        q_start = np.quaternion(start_ori[0], start_ori[1], start_ori[2], start_ori[3])
        print("q_start", q_start)
        q_goal = np.quaternion(goal_pose.pose.orientation.w, goal_pose.pose.orientation.x, goal_pose.pose.orientation.y,
                               goal_pose.pose.orientation.z)
        print("q_goal", q_goal)
        inner_prod = q_start.x * q_goal.x + q_start.y * q_goal.y + q_start.z * q_goal.z + q_start.w * q_goal.w
        if inner_prod < 0:
            q_start.x = -q_start.x
            q_start.y = -q_start.y
            q_start.z = -q_start.z
            q_start.w = -q_start.w
        inner_prod = q_start.x * q_goal.x + q_start.y * q_goal.y + q_start.z * q_goal.z + q_start.w * q_goal.w
        theta = np.arccos(np.abs(inner_prod))
        print(theta)
        interp_dist_polar = 0.001/2
        step_num_polar = math.floor(theta / interp_dist_polar)

        print("num of steps polar", step_num_polar)

        step_num = np.max([step_num_polar, step_num_lin])

        print("num of steps", step_num)
        x = np.linspace(start[0], goal_pose.pose.position.x, step_num)
        y = np.linspace(start[1], goal_pose.pose.position.y, step_num)
        z = np.linspace(start[2], goal_pose.pose.position.z, step_num)

        step_num = int(step_num)

        goal = PoseStamped()
        goal.header.frame_id = "panda_link0"

        goal.pose.position.x = x[0]
        goal.pose.position.y = y[0]
        goal.pose.position.z = z[0]

        quat = np.slerp_vectorized(q_start, q_goal, 0.0)
        goal.pose.orientation.x = quat.x
        goal.pose.orientation.y = quat.y
        goal.pose.orientation.z = quat.z
        goal.pose.orientation.w = quat.w
        self.goal_pub.publish(goal)


        goal = PoseStamped()
        for i in range(step_num):

            now = time.time()
            goal.header.seq = 1
            goal.header.stamp = rospy.Time.now()
            goal.header.frame_id = "panda_link0"

            goal.pose.position.x = x[i]
            goal.pose.position.y = y[i]
            goal.pose.position.z = z[i]
            quat = np.slerp_vectorized(q_start, q_goal, i / step_num)
            goal.pose.orientation.x = quat.x
            goal.pose.orientation.y = quat.y
            goal.pose.orientation.z = quat.z
            goal.pose.orientation.w = quat.w
            
            self.goal_pub.publish(goal)
            self.r.sleep()

    def box_pose_from_2_points(self, point1, rot1, point2, rot2):

        new_points = np.zeros((4,3))
        xaxis = np.array([0,0.5,0])
        print(rot2)
        quaternion1 = Quaternion(rot1[0], rot1[1], rot1[2], rot1[3])
        quaternion2 = Quaternion(rot2[0], rot2[1], rot2[2], rot2[3])

        new_points[0] = np.array(point1) + quaternion1.rotate(xaxis)
        new_points[1] = np.array(point1) - quaternion1.rotate(xaxis)
        new_points[2] = np.array(point2) + quaternion2.rotate(xaxis)
        new_points[3] = np.array(point2) - quaternion2.rotate(xaxis)

        new_points[:,2] = 0.2

        y_axis_new = new_points[1] - new_points[0]

        ori = quaternion.from_euler_angles(np.array([0, 0, np.arctan2(y_axis_new[1], y_axis_new[0])]))

        # Rotate resulting orientation by 90 degrees to match our convention of the box frame
        ori = np.quaternion(np.sqrt(0.5), 0, 0, -np.sqrt(0.5)) * ori

        l1 = Line(new_points[0], new_points[1])
        l2 = Line(new_points[2], new_points[3])

        intersect = np.array(l1.intersection(l2))[0]

        pose = PoseStamped()
        pose.pose.position.x = intersect[0]
        pose.pose.position.y = intersect[1]
        pose.pose.position.z = intersect[2]
        pose.pose.orientation.w = ori.w
        pose.pose.orientation.x = ori.x
        pose.pose.orientation.y = ori.y
        pose.pose.orientation.z = ori.z
        
        return pose

if __name__ == '__main__':
    offline = False
    
    rospy.init_node('BoardDetector', anonymous=True)
    detector = BoardDetector()
    time.sleep(1)
    # Add line with request to the pointcloud server to fill out pose_icp
    detector.execute_touch(offline)
    print('two collision poses are\n', detector.coll_points_to_save, detector.coll_oris_to_save)
    print('box pose is\n', detector.pose_box)

    transform_ref = np.load("transform_box_ref.npy")
    transform_new = pose_2_transformation(detector.pose_box)
    transform_ref_2_new = transform_new @ np.linalg.inv(transform_ref)
    detector.pose_ref_2_new = transformation_2_pose(transform_ref_2_new)

    while not rospy.is_shutdown():
        detector.pose_ref_2_new_pub.publish(detector.pose_ref_2_new)
        rospy.sleep(1)

