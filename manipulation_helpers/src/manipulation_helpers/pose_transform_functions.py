import numpy as np
import quaternion #numpy quaternion
from geometry_msgs.msg import PoseStamped, Pose

def orientation_2_quaternion(orientation):
    return np.quaternion(orientation.w, orientation.x, orientation.y, orientation.z)

def position_2_array(position):
    return np.array([position.x, position.y, position.z])

def pose_2_transformation(pose_st: PoseStamped):
    quaternion_orientation = orientation_2_quaternion(pose_st.pose.orientation)
    translation = position_2_array(pose_st.pose.position)
    rotation_matrix = quaternion.as_rotation_matrix(quaternion_orientation)
    transformation_matrix = np.identity(4)
    transformation_matrix[0:3, 0:3] = rotation_matrix
    transformation_matrix[0:3, 3] = translation
    return transformation_matrix

def array_quat_2_pose(pos_array, quat):
    pose_st = PoseStamped()
    pose_st.pose.position.x = pos_array[0]
    pose_st.pose.position.y = pos_array[1]
    pose_st.pose.position.z = pos_array[2]
    pose_st.pose.orientation.x = quat.x
    pose_st.pose.orientation.y = quat.y
    pose_st.pose.orientation.z = quat.z
    pose_st.pose.orientation.w = quat.w
    return pose_st

def transformation_2_pose(transformation_matrix):
    pos_array = transformation_matrix[0:3, 3]
    rotation_matrix = transformation_matrix[0:3, 0:3]
    quat = quaternion.from_rotation_matrix(rotation_matrix)
    pose_st = array_quat_2_pose(pos_array, quat)
    return pose_st

