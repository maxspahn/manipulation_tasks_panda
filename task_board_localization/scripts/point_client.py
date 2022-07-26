#!/usr/bin/env python

import sys
import rospy
from icp_board_detector.srv import PointDetect, PointDetectRequest, PointDetectResponse

from geometry_msgs.msg import Pose
from sensor_msgs.msg import PointCloud2

def point_detect_client():
    '''
    running of the perception service client
    @return:
    '''
    print("check")
    rate = rospy.Rate(1)

    try:
        # Obtain a single pointcloud2
        point_cloud = rospy.wait_for_message("/camera/depth/color/points", PointCloud2)
        point_detect = rospy.ServiceProxy("/point_detect", PointDetect)
        rospy.loginfo("WAITING FOR SERVICE")

        # Send point cloud the server
        rospy.wait_for_service("/point_detect",5)
        orientation = point_detect(point_cloud)
        rospy.loginfo("retrieved iterative closest point")
        rate.sleep()
        return PointDetectResponse(orientation)
    except rospy.ServiceException as e:
        print("Service call failed")
    return


if __name__ == "__main__":
    rospy.init_node("point_client")

    resp = point_detect_client()
    trans_rot_pub = rospy.Publisher("/trans_rot", Pose, queue_size=0)
    while not rospy.is_shutdown():
        trans_rot_pub.publish(resp.pose.pose)
    print(resp)
    print(resp)
    rospy.spin()



