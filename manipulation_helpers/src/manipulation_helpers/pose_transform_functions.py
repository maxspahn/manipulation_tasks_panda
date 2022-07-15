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
