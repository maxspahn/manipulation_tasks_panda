#!/usr/bin/env python3.6
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
from manipulation_helpers.pose_transform_functions import orientation_2_quaternion, pose_2_transformation, position_2_array

class BoardDetector():
    '''
    a wrapper class for board detection
    '''
    def __init__(self):
        self.r=rospy.Rate(100)
        self.sub = rospy.Subscriber('/cartesian_pose', PoseStamped, self.update_pose)
        self.force_feedback_sub = rospy.Subscriber('/force_torque_ext', WrenchStamped, self.force_feedback_checker)
        self.joint_states_sub = rospy.Subscriber("/joint_states", JointState, self.joint_states_callback)
        self.goal_pub = rospy.Publisher('/equilibrium_pose', PoseStamped, queue_size=0)
        self.configuration_pub = rospy.Publisher("/equilibrium_configuration", Float32MultiArray, queue_size=0)
        self.ref_to_new_pose_pub = rospy.Publisher('/pose_ref_to_new', PoseStamped, queue_size=0)
        self.ref_to_new_pose = PoseStamped()
        self.collided = False
        self.board_width = 0.16
        self.board_length = 0.26
        self.board_height = 0.07
        self.clearance = 0.1
        self.detected = False
        self.force_feedback = None
        self.torque_z = None
        self.curr_pos = np.empty([3])
        self.curr_ori = np.empty([3])
        self.trajx = []
        self.trajy = []
        self.trajz = []
        self.orix = []
        self.oriy = []
        self.coll_points = [np.array([[0, 0, 0]])]
        self.coll_points_to_save = []
        self.coll_oris = [np.array([[0, 0, 0, 0]])]
        self.coll_oris_to_save = []
        self.det_box_frame = 'test_1'
        self.force_threshold = 7
        self.K_pos = 600
        self.K_ori = 50
        self.K_nul = 5.0

        self.flipped_case = False
        self.which_case = [0,0]

        self.trans_old = [0.3942, 0.1512, 0.16]
        self.rot_old = np.quaternion(np.sqrt(0.5), 0, 0, -np.sqrt(0.5)) * np.quaternion(0.9999, 0, 0, 0.0111)

        self.pose_icp = None
        self.offset = 0.01

    def set_stiffness(self, k_t1, k_t2, k_t3, k_r1, k_r2, k_r3, k_ns):

        set_K = dynamic_reconfigure.client.Client('/dynamic_reconfigure_compliance_param_node', config_callback=None)
        set_K.update_configuration({"translational_stiffness_X": k_t1})
        set_K.update_configuration({"translational_stiffness_Y": k_t2})
        set_K.update_configuration({"translational_stiffness_Z": k_t3})
        set_K.update_configuration({"rotational_stiffness_X": k_r1})
        set_K.update_configuration({"rotational_stiffness_Y": k_r2})
        set_K.update_configuration({"rotational_stiffness_Z": k_r3})
        set_K.update_configuration({"nullspace_stiffness": k_ns})

    def set_configuration(self, joint):
        joint_des = Float32MultiArray()
        joint_des.data = np.array(joint).astype(np.float32)
        self.configuration_pub.publish(joint_des)

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

    def trans_icp_callback(self, data):
        self.pose_icp = data

    # TODO this would be much better in a config file since the points are now fixed
    # points defined with respect to local initially detected base frame
    def get_trajectories(self):
        # For x-axis approach we just want the orientation to be aligned with local frame
        self.orix = np.array([1, 0, 0, 0])
        # For y-axis orientation just a 90 deg rotation about z-axis to touch with same flat side of flange
        # These orientations are fixed throughout approach
        self.oriy = np.array([np.sqrt(0.5), 0, 0, -np.sqrt(0.5)])


        # I'm assuming we are putting the coordinate systems in a corner,
        # and that x-axis goes along length and y along width

        # Point outside and above box
        p1x = [-self.clearance, self.board_width/2, self.clearance]
        p2x = [p1x[0], p1x[1], -self.board_height*0.6]
        p3x = [self.board_width/2, p2x[1], p2x[2]]

        self.trajx = [p1x, p2x, p3x]

        p1y = [self.board_length*0.7, -self.clearance, self.clearance]
        p2y = [p1y[0], p1y[1], -self.board_height*0.7]
        p3y = [p2y[0], self.board_width * 0.35, p2y[2]]

        self.trajy = [p1y, p2y, p3y]

        p1z = [self.board_length/2, self.board_width/2, self.clearance]
        p2z = [self.board_length/2, self.board_width/2, -self.board_height/2]

        self.trajz = [p1z, p2z]

        return

    # TODO This can be a callback for a subscriber to a topic where the initially detected box is detected
    #  so that it firsts sets the trans and rot (from the transform from the topic) and executes the touches

    def execute_touch_ref(self):
        self.get_trajectories()
        K_z = 1000

        # Note stiffness in rotation about z is set to zero in last loop to allow gripper to 'straighten'
        for i in range(len(self.trajx)):
            print("Kpos", self.K_pos)
            if i == 2:
                self.set_stiffness(self.K_pos, self.K_pos, K_z, self.K_ori, self.K_ori, 0.0, 0)
            else:
                self.set_stiffness(self.K_pos, self.K_pos, K_z, self.K_ori, self.K_ori, self.K_ori, 5.0)

            if self.collided:
                break
            goal = self.point_quat_to_goal_ref_to_base(self.trajx[i], self.orix, self.trans_old, self.rot_old)
            self.go_to_pose(goal)

        self.coll_points_to_save.append(self.coll_points[-1])
        self.coll_oris_to_save.append(self.coll_oris[-1])

        # TODO consider not breaking when collision occurs, just recording the points?

        goal = self.point_quat_to_goal_ref_to_base(self.trajx[0], self.orix, self.trans_old, self.rot_old)
        self.go_to_pose(goal)

        for i in range(len(self.trajy)):
            if i == 2: self.set_stiffness(self.K_pos, self.K_pos, K_z, self.K_ori, self.K_ori, 0, 0)
            else:      self.set_stiffness(self.K_pos, self.K_pos, K_z, self.K_ori, self.K_ori, self.K_ori, 5.0)

            if self.collided:
                break
            goal = self.point_quat_to_goal_ref_to_base(self.trajy[i], self.oriy, self.trans_old, self.rot_old)
            self.go_to_pose(goal)

        self.coll_points_to_save.append(self.coll_points[-1])
        self.coll_oris_to_save.append(self.coll_oris[-1])

        goal = self.point_quat_to_goal_ref_to_base(self.trajy[0], self.oriy, self.trans_old, self.rot_old)
        self.go_to_pose(goal)

        np.save("coll_points_ref", self.coll_points_to_save)
        np.save("coll_oris_ref", self.coll_oris_to_save)

        # for i in range(len(self.trajz)):
        #     if self.collided:
        #         break
        #     goal = self.point_quat_to_goal(self.trajz[i], self.oriy, self.trans, self.rot)
        #     self.go_to_pose(goal)

        # self.coll_points_to_save = np.append(self.coll_points_to_save, self.coll_points[-1])
        # self.coll_oris_to_save = np.append(self.coll_oris_to_save, self.coll_oris[-1])

        # goal = self.point_quat_to_goal(self.trajz[0], self.oriy, self.trans, self.rot)
        # self.go_to_pose(goal)
        # np.save("coll_points2", self.coll_points_to_save.transpose())
        # np.save("coll_oris1", self.coll_oris_to_save.transpose())

        return
    
    def execute_touch(self):
        self.get_trajectories()
        # Note stiffness in rotation about z is set to zero in last loop to allow gripper to 'straighten'
        for i in range(len(self.trajx)):
            self.flipped_case = False
            if i == 2: self.set_stiffness(self.K_pos, self.K_pos, self.K_pos, self.K_ori, self.K_ori, 0.0, 0.0)
            else:      self.set_stiffness(self.K_pos, self.K_pos, self.K_pos, self.K_ori, self.K_ori, self.K_ori, 0.0)

            # if self.collided:
            #     break
            goal = self.point_quat_to_goal_ref_to_base(self.trajx[i], self.orix, self.trans_old, self.rot_old)
            goal = self.point_quat_to_goal_new_box(goal, self.trans, self.rot)
            self.go_to_pose(goal)
            if self.flipped_case:
                self.which_case[0] = 1
                print("x collision is now flipped")
                self.orix = np.quaternion(0,0,0,1) * self.orix

        self.coll_points_to_save.append(self.coll_points[-1])
        self.coll_oris_to_save.append(self.coll_oris[-1])

        # TODO consider not breaking when collision occurs, just recording the points?

        goal = self.point_quat_to_goal_ref_to_base(self.trajx[0], self.orix, self.trans_old, self.rot_old)
        goal = self.point_quat_to_goal_new_box(goal, self.trans, self.rot)
        self.go_to_pose(goal)

        for i in range(len(self.trajy)):
            if i == 2: self.set_stiffness(self.K_pos, self.K_pos, self.K_pos, self.K_ori, self.K_ori, 0, 0.0)
            else:      self.set_stiffness(self.K_pos, self.K_pos, self.K_pos, self.K_ori, self.K_ori, self.K_ori, 0.0)

            # if self.collided:
            #     break
            goal = self.point_quat_to_goal_ref_to_base(self.trajy[i], self.oriy, self.trans_old, self.rot_old)
            goal = self.point_quat_to_goal_new_box(goal, self.trans, self.rot)
            self.go_to_pose(goal)
            if self.flipped_case:
                self.which_case[1] = 1
                print("y collision is now flipped")
                self.oriy = np.quaternion(0,0,0,1) * self.oriy

        self.coll_points_to_save.append(self.coll_points[-1])
        self.coll_oris_to_save.append(self.coll_oris[-1])
        goal = self.point_quat_to_goal_ref_to_base(self.trajy[0], self.oriy, self.trans_old, self.rot_old)
        goal = self.point_quat_to_goal_new_box(goal, self.trans, self.rot)
        self.go_to_pose(goal)

        np.save("coll_points_new", self.coll_points_to_save)
        np.save("coll_oris_new", self.coll_oris_to_save)

        # for i in range(len(self.trajz)):
        #     # if self.collided:
        #     #     break
        #     goal = self.point_quat_to_goal(self.trajz[i], self.oriy, self.trans, self.rot)
        #     self.go_to_pose(goal)

        # self.coll_points_to_save = np.append(self.coll_points_to_save, self.coll_points[-1])
        # self.coll_oris_to_save = np.append(self.coll_oris_to_save, self.coll_oris[-1])

        # # point_quat_to_goal converts to base frame
        # goal = self.point_quat_to_goal_ref_to_base(self.trajz[0], self.oriy, self.trans, self.rot)
        # goal = self.point_quat_to_goal_new_box
        # goal = transform_cam 

        # self.go_to_pose(goal)
        # np.save("coll_points2", self.coll_points_to_save.transpose())
        # np.save("coll_oris1", self.coll_oris_to_save.transpose())

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
            if self.curr_joint[6] > 3:
                desired_joints = self.curr_joint
                desired_joints[6] = desired_joints[6] - np.pi
                self.set_configuration(desired_joints)
                self.set_stiffness(self.K_pos, self.K_pos, self.K_pos, self.K_ori, self.K_ori, 0, 5)
                rospy.sleep(5.0)
                self.set_stiffness(self.K_pos, self.K_pos, self.K_pos, self.K_ori, self.K_ori, self.K_ori, 5)
                self.flipped_case = True
                q_goal = np.quaternion(0, 0, 0, 1) * q_goal
            elif self.curr_joint[6] < 3:
                desired_joints = self.curr_joint
                desired_joints[6] = desired_joints[6] + np.pi
                self.set_configuration(desired_joints)
                self.set_stiffness(self.K_pos, self.K_pos, self.K_pos, self.K_ori, self.K_ori, 0, 5)
                rospy.sleep(5.0)
                self.set_stiffness(self.K_pos, self.K_pos, self.K_pos, self.K_ori, self.K_ori, self.K_ori, 5)
                self.flipped_case = True
                q_goal = np.quaternion(0, 0, 0, 1) * q_goal

            # if self.collided:
            #     break
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

    # Takes a pose (as a point and quaternion orientation in arrays), converts to PoseStamped msg, and transforms
    # from your source_frame (specified by trans and rot) to panda_link0
    def point_quat_to_goal_ref_to_base(self, point, ori, trans, rot):

        goal = PoseStamped()

        quat_ori = np.quaternion(ori[0], ori[1], ori[2], ori[3])
        # Converting the quaternion to rotation matrix, to make a homogenous transformation and transform the points
        # with one matrix operation 'transform @ point'
        rot_matrix = quaternion.as_rotation_matrix(rot)
        transform = np.append(rot_matrix, [[0, 0, 0]], axis=0)
        transform = np.append(transform, [[trans[0]], [trans[1]], [trans[2]], [1]], axis=1)
        point = np.array([point[0], point[1], point[2], 1])
        new_point = transform @ point

        # Note 'extra' final rotation by q(0, 1, 0, 0) (180 deg about x axis) since we want gripper facing down
        new_ori = np.quaternion(0, 0, 0, 1) * np.quaternion(0, 1, 0, 0) * rot * quat_ori
        print("new_ori", new_ori)
        goal.pose.position.x = new_point[0]
        goal.pose.position.y = new_point[1]
        goal.pose.position.z = new_point[2]

        goal.pose.orientation.w = new_ori.w
        goal.pose.orientation.x = new_ori.x
        goal.pose.orientation.y = new_ori.y
        goal.pose.orientation.z = new_ori.z

        return goal

    def point_quat_to_goal_new_box(self, goal, trans, rot):

        point = [goal.pose.position.x, goal.pose.position.y, goal.pose.position.z, 1]
        quat_ori = np.quaternion(goal.pose.orientation.w, goal.pose.orientation.x,
                                goal.pose.orientation.y, goal.pose.orientation.z)

        rot_matrix = quaternion.as_rotation_matrix(rot)
        transform_icp = np.append(rot_matrix, [[0, 0, 0]], axis=0)
        transform_icp = np.append(transform_icp, [[trans[0]], [trans[1]], [trans[2]], [1]], axis=1)

        # transform_cam = self.tfbuffer.lookup_transform('panda_link0', 'camera_depth_optical_frame_static', rospy.Time.now(), rospy.Duration(1.0))

        trans_cam = np.array([0.483, 0.021,
                                0.58])
        quat_cam = np.quaternion(0.006, 0.734, -0.679, 0.006)
        rot_matrix_cam = quaternion.as_rotation_matrix(quat_cam)
        transform_matrix_cam = np.append(rot_matrix_cam, [[0, 0, 0]], axis=0)
        transform_matrix_cam = np.append(transform_matrix_cam, [[trans_cam[0]], [trans_cam[1]], [trans_cam[2]], [1]], axis=1)
        # Converting the quaternion to rotation matrix, to make a homogenous transformation and transform the points
        # with one matrix operation 'transform @ point'

        point1 = np.linalg.inv(transform_matrix_cam) @ point
        # print('first ', point1)
        # point1_other = transform_matrix_cam @ point
        # print('second ', point1_other)
        point2 = transform_icp @ point1
        new_point = transform_matrix_cam @ point2

        new_ori = quat_cam.inverse() * rot * quat_cam * quat_ori

        goal = PoseStamped()

        goal.pose.position.x = new_point[0]
        goal.pose.position.y = new_point[1]
        goal.pose.position.z = new_point[2]

        goal.pose.orientation.w = new_ori.w
        goal.pose.orientation.x = new_ori.x
        goal.pose.orientation.y = new_ori.y
        goal.pose.orientation.z = new_ori.z

        return goal

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

        # __import__('pdb').set_trace()

        rot_matrix = quaternion.as_rotation_matrix(ori)

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

    def ref_to_new_transf_pose(self, pose_ref, pose_new):
        trans_new = [pose_new.pose.position.x, pose_new.pose.position.y, pose_new.pose.position.z]
        quat_new = np.quaternion(pose_new.pose.orientation.w, pose_new.pose.orientation.x, pose_new.pose.orientation.y,
                                 pose_new.pose.orientation.z)

        rot_matrix_new = quaternion.as_rotation_matrix(quat_new)
        transform_new = np.identity(4)
        transform_new[0:3, 0:3] = rot_matrix_new
        transform_new[0:3, 3] = trans_new

        trans_ref = [pose_ref.pose.position.x, pose_ref.pose.position.y, pose_ref.pose.position.z]
        quat_ref = np.quaternion(pose_ref.pose.orientation.w, pose_ref.pose.orientation.x, pose_ref.pose.orientation.y,
                                 pose_ref.pose.orientation.z)

        rot_matrix_ref = quaternion.as_rotation_matrix(quat_ref)
        transform_ref = np.identity(4)
        transform_ref[0:3, 0:3] = rot_matrix_ref
        transform_ref[0:3, 3] = trans_ref

        trans_ref_to_new = transform_new @ np.linalg.inv(transform_ref)
        quat_to_new = quaternion.from_rotation_matrix(trans_ref_to_new[0:3, 0:3])

        pose_ref_to_new = PoseStamped()
        pose_ref_to_new.pose.position.x = trans_ref_to_new[0,3]
        pose_ref_to_new.pose.position.y = trans_ref_to_new[1,3]
        pose_ref_to_new.pose.position.z = trans_ref_to_new[2,3]
        pose_ref_to_new.pose.orientation.x = quat_to_new.x
        pose_ref_to_new.pose.orientation.y = quat_to_new.y
        pose_ref_to_new.pose.orientation.z = quat_to_new.z
        pose_ref_to_new.pose.orientation.w = quat_to_new.w

        return pose_ref_to_new

if __name__ == '__main__':
    offline = False
    
    rospy.init_node('BoardDetector', anonymous=True)
    detector = BoardDetector()
    time.sleep(1)
    # Only does the reference poses with fixed axes to get reference points
    if offline: 
        detector.execute_touch_ref()
        print('two collision poses are', detector.coll_points_to_save, detector.coll_oris_to_save)
        pose_box_ref = detector.box_pose_from_2_points(detector.coll_points_to_save[0], detector.coll_oris_to_save[0],
                                                    detector.coll_points_to_save[1], detector.coll_oris_to_save[1])
        print('ref_pose is', pose_box_ref)


    if not offline:
        trans_icp_sub = rospy.Subscriber('/trans_rot', Pose, detector.trans_icp_callback)
        rospy.sleep(1.0)
        detector.trans = np.array([detector.pose_icp.position.x, detector.pose_icp.position.y, detector.pose_icp.position.z])
        detector.rot  = np.quaternion(detector.pose_icp.orientation.w, detector.pose_icp.orientation.x, detector.pose_icp.orientation.y, detector.pose_icp.orientation.z)
        detector.execute_touch()
        print('two collision poses are', detector.coll_points_to_save, detector.coll_oris_to_save)
        pose_box_new = detector.box_pose_from_2_points(detector.coll_points_to_save[0], detector.coll_oris_to_save[0],
                                                       detector.coll_points_to_save[1], detector.coll_oris_to_save[1])
        print('new_pose is', pose_box_new)

        coll_points_ref = np.load('coll_points_ref_1850.npy')
        coll_oris_ref = np.load('coll_oris_ref_1850.npy')
        pose_box_ref = detector.box_pose_from_2_points(coll_points_ref[0], coll_oris_ref[0], coll_points_ref[1], coll_oris_ref[1])
        detector.ref_to_new_pose = detector.ref_to_new_transf_pose(pose_box_ref, pose_box_new)
        print(detector.ref_to_new_pose)
    while not rospy.is_shutdown():
        detector.ref_to_new_pose_pub.publish(detector.ref_to_new_pose)
        rospy.sleep(1)

