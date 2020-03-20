#!/usr/bin/env python3
import os
import cv2
import numpy as np
import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

class stereoMatcher:
	def __init__(self, stereo_mode, topics_to, camera_info, baseline):
		# Read focal length
		self._FOCAL_LENGTH = camera_info.K[0]
		# Read Cx and Cy
		self._CX = camera_info.K[2]
		self._CY = camera_info.K[5]
		# Read baseline distance
		self._STEREO_BASELINE = baseline

		self._BLOCK_SIZE = 5
		self._MIN_DISPARITY = 0
		self._NUM_DISPARITIES = 80
		self._LAMBDA = 80000
		self._SIGMA = 1.8

		# StereoBM
		if stereo_mode == "StereoBM":
			self._left_stereo_matcher = cv2.StereoBM_create()
			self._left_stereo_matcher.setMinDisparity(self._MIN_DISPARITY)
			self._left_stereo_matcher.setNumDisparities(self._NUM_DISPARITIES)
			self._left_stereo_matcher.setBlockSize(self._BLOCK_SIZE)
			self._left_stereo_matcher.setSpeckleRange(32)
			self._left_stereo_matcher.setSpeckleWindowSize(100)
			self._left_stereo_matcher.setUniquenessRatio(15)
			self._left_stereo_matcher.setDisp12MaxDiff(1)

		# StereoSGBM
		elif stereo_mode == "StereoSGBM":
			# Left stereo matcher
			self._left_stereo_matcher = cv2.StereoSGBM_create(minDisparity=self._MIN_DISPARITY,
			numDisparities=self._NUM_DISPARITIES,
			blockSize=self._BLOCK_SIZE,
			P1=8 * 3 * self._BLOCK_SIZE ** 2,
			P2=32 * 3 * self._BLOCK_SIZE ** 2,
			disp12MaxDiff=1,
			uniquenessRatio=15,
			speckleWindowSize=100,
			speckleRange=32,
			preFilterCap=63,
			mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

		# Right stereo matcher
		self._right_stereo_matcher = cv2.ximgproc.createRightMatcher(self._left_stereo_matcher)
		# Create WLS filter
		self._wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self._left_stereo_matcher)
		self._wls_filter.setLambda(self._LAMBDA)
		self._wls_filter.setSigmaColor(self._SIGMA)

		self._bridge = CvBridge()
		# Create Colored disparity publisher
		self._disparity_publisher = rospy.Publisher(topics_to['disparity'], Image, queue_size=2)
		# Create depth map publisher
		self._depthmap_publisher = rospy.Publisher(topics_to['depth'], Image, queue_size=2)


	def calculate_disparity(self, left_image, right_image):
		# Change left and right frames to gray scale
		gray_left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
		gray_right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)
		# Calculate disparity
		left_disparity = self._left_stereo_matcher.compute(gray_left_image, gray_right_image)
		right_disparity = self._right_stereo_matcher.compute(gray_right_image, gray_left_image)
		# Filter disparity map
		filtered_disparity = self._wls_filter.filter(left_disparity, gray_left_image, None,
		right_disparity).astype(np.float32) / 16

		return filtered_disparity

	def calculate_depth(self, disparity):
		disparity_map = disparity.copy()
		disparity_map[disparity_map == 0] = 0.1
		disparity_map[disparity_map == -1] = 0.1
		depth_map = np.ones(disparity_map.shape, np.single)
		depth_map[:] = self._FOCAL_LENGTH * self._STEREO_BASELINE / disparity_map[:]
		return depth_map

	def display_disparity(self, left_camera_data, right_camera_data):
		try:
			left_frame = self._bridge.imgmsg_to_cv2(left_camera_data, "rgb8")
			right_frame = self._bridge.imgmsg_to_cv2(right_camera_data, "rgb8")
		except CvBridgeError as e:
       			print(e)

		# Calculate disparity map
		disparity = self.calculate_disparity(left_frame, right_frame)
		# Normalize disparity map
		normalized_disparity = np.uint8(cv2.normalize(src=disparity, dst=None, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX))
		# Create colored disparity map
		colored_disparity = cv2.applyColorMap(normalized_disparity, cv2.COLORMAP_RAINBOW)
		# Calculate depth map
		depth_map = self.calculate_depth(disparity)
		# Publish disparity map

		try:        
			self._disparity_publisher.publish(self._bridge.cv2_to_imgmsg(colored_disparity, encoding="rgb8"))
			self._depthmap_publisher.publish(self._bridge.cv2_to_imgmsg(depth_map))
			rospy.loginfo("Published disparity & depth map")
		except CvBridgeError as e:
			print(e)



def main():
	rospy.init_node('stereovision')

	STEREO_MODE = rospy.get_param('stereo_mode', "StereoSGBM")
	CAMERA_BASELINE = rospy.get_param('baseline', 0.4)

	TOPICS_FROM = {'left_camera':"/prius/left_camera/image_raw", 'right_camera':"/prius/right_camera/image_raw"}
	TOPICS_TO = {'depth':"stereovision/depth_map", 'disparity':"stereovision/disparity"}
	
	LEFT_CAMERA_INFO = rospy.wait_for_message('/prius/right_camera/camera_info', CameraInfo)
	# RIGHT_CAMERA_INFO = rospy.wait_for_message('/prius/right_camera/camera_info', CameraInfo)

	# Suppose both cameras are matched so send info of left camera only
	stereo_matcher = stereoMatcher(STEREO_MODE, TOPICS_TO, LEFT_CAMERA_INFO, CAMERA_BASELINE)
	left_camera_subscriber = message_filters.Subscriber(TOPICS_FROM['left_camera'], Image, buff_size=2**24, queue_size=2)
	right_camera_subscriber = message_filters.Subscriber(TOPICS_FROM['right_camera'], Image, buff_size=2**24, queue_size=2)
	time_synchronizer = message_filters.ApproximateTimeSynchronizer([left_camera_subscriber, right_camera_subscriber], 2, 0.05)
	time_synchronizer.registerCallback(stereo_matcher.display_disparity)
	
	rate = rospy.Rate(20)
	try:
		while not rospy.is_shutdown():
			rate.sleep()
			rospy.spin()

	except rospy.ROSInterruptException:
		rospy.loginfo("Shutting down")

if __name__ == "__main__":
	main()
