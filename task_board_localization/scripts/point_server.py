#!/usr/bin/env python

import rospy
import actionlib

import roslib
import time
import quaternion

import open3d as o3d
import numpy as np
import copy
import ros_numpy
from geometry_msgs.msg import Pose

import sys
sys.path.append("../../")
from icp_board_detector.srv import PointDetect, PointDetectRequest, PointDetectResponse

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size #* 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=100))
    o3d.visualization.draw_geometries([pcd_down],
                                  point_show_normal=True)
    radius_feature = voxel_size #* 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size, target):
    print(":: Load two point clouds and disturb initial pose.")
    source = o3d.io.read_point_cloud("/home/richard/lxj/TU_Delft/erf2022/catkin_ws_LfD/src/ERF2022_TUDelft/icp_board_detector/scripts/final1.ply")
    model_numpy = ros_numpy.numpify(target)

    xyz = [(x, y, z) for x, y, z, rgb in model_numpy]  # (why cannot put this line below rgb?)
    # model_numpy = model_numpy.T
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(xyz))
    draw_registration_result(source, pcd, np.identity(4))

    # processing of the target pointcloud
    # get the bounds of the initial point cloud
    lb = pcd.get_min_bound() # (3,)
    ub = pcd.get_max_bound()
    print("lower bounds of the un-processed pointcloud: ", lb)
    print("lower bounds of the un-processed pointcloud: ", ub)
    # cropping both source and target along z-axis
    ub[-1] = 0.45
    min_bound = (lb)#np.array([0.1, 0.1, 0.1]).reshape([3, 1])
    max_bound = (ub)#np.array([0.2, 0.2, 0.2]).reshape([3, 1])
    box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    target = pcd.crop(box)
    source = source.crop(box)

    # o3d.io.write_point_cloud("/home/richard/lxj/TU_Delft/erf2022/catkin_ws_LfD/src/ERF2022_TUDelft/icp_board_detector/scripts/final1.ply", target)
    # pcd_load = o3d.io.read_point_cloud("target.ply")
    # trans_init = np.asarray([[0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0],
    #                          [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    # source.transform(trans_init)
    draw_registration_result(source, target, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)

    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=source_down, target=target_down, source_feature=source_fpfh, target_feature=target_fpfh, max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False), ransac_n=4, checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.9999999), mutual_filter=False)
    return result

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    target.estimate_normals()
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

# gives the point cloud of the camera sensor (scene)
def point_detect(req):
    voxel_size = 0.008  # means 5cm for the dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(voxel_size, req.cloud)

    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    draw_registration_result(source_down, target_down, result_ransac.transformation)
    result_icp = refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, result_ransac)

    trans = result_icp.transformation
    draw_registration_result(source, target, result_icp.transformation)
    rot_matrix = trans[:3, :3]
    translation = trans[:, 3]

    # Store the created homogeneous matrix as a Pose message
    detected_quat = quaternion.from_rotation_matrix(rot_matrix)
    detected_pose = Pose()
    detected_pose.position.x = translation[0]
    detected_pose.position.y = translation[1]
    detected_pose.position.z = translation[2]
    detected_pose.orientation.w = detected_quat.w
    detected_pose.orientation.x = detected_quat.x
    detected_pose.orientation.y = detected_quat.y
    detected_pose.orientation.z = detected_quat.z

    return PointDetectResponse(detected_pose)
    
def point_server():
    s = rospy.Service("point_detect", PointDetect, point_detect)    

if __name__=="__main__":
    rospy.init_node("point_server")
    point_server()
    print("point server started")
    rospy.spin()
