#!/usr/bin/env python3

import rospy
import os
import numpy as np
import cv2
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from graduation_project_simulation.msg import detectedObject, detectionMsg

class stereoMatcher:
	def __init__(self, stereo_mode, topics_to, camera_info, baseline):
		# Read focal length
		self._FOCAL_LENGTH = camera_info.K[0]
		# Read Cx and Cy
		self._CX = camera_info.K[2]
		self._CY = camera_info.K[5]
		# Read baseline distance
		self._STEREO_BASELINE = baseline
		# Read width and height
		self._camera_width = camera_info.width
		self._camera_height = camera_info.height

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
			mode=cv2.STEREO_SGBM_MODE_HH4)

		# Right stereo matcher
		self._right_stereo_matcher = cv2.ximgproc.createRightMatcher(self._left_stereo_matcher)
		# Create WLS filter
		self._wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self._left_stereo_matcher)
		self._wls_filter.setLambda(self._LAMBDA)
		self._wls_filter.setSigmaColor(self._SIGMA)

		self._bridge = CvBridge()

		self._left_frame = None
		self._right_frame = None
		# Create Colored disparity publisher
		self._object_distance_frame_publisher = rospy.Publisher(topics_to['frame'], Image, queue_size=1)
		self._object_distance_publisher = rospy.Publisher(topics_to['msg'], detectionMsg, queue_size=1)
		self._processing = False
		self._new_msg = False

	def calculate_disparity(self, left_image, right_image):
		# Change left and right frames to gray scale
		gray_left_image = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
		gray_right_image = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)
		# Calculate disparity
		left_disparity = self._left_stereo_matcher.compute(gray_left_image, gray_right_image)
		right_disparity = self._right_stereo_matcher.compute(gray_right_image, gray_left_image)
		# Filter disparity map
		filtered_disparity = self._wls_filter.filter(left_disparity, gray_left_image, None, right_disparity).astype(np.float32) / 16
		return filtered_disparity


	def calculate_point_cloud(self, required_point, disparity):
		z = self._FOCAL_LENGTH * self._STEREO_BASELINE / disparity
		x = z * (required_point[1] - self._CX) / self._FOCAL_LENGTH
		y = z * (required_point[0] - self._CY) / self._FOCAL_LENGTH
		return (x, y, z)

	def read_frames(self, left_camera_data, right_camera_data):
		# Check if the frames were processed
		if not self._processing:
			try:
				self._left_frame = self._bridge.imgmsg_to_cv2(left_camera_data, "rgb8")
				self._right_frame = self._bridge.imgmsg_to_cv2(right_camera_data, "rgb8")
				rospy.loginfo("Received camera frames")
				self._new_msg = True
			except CvBridgeError as e:
	       			print(e)

	

	def display_disparity(self, detections_msg):
		# Check if new frames were received
		if self._new_msg:
			self._processing = True
			self._new_msg = False

			left_frame_copy = self._left_frame.copy()
			# Measure distance to objects
			for detected_object in detections_msg.detectedObjects:
				new_left = min(max(0, detected_object.Center[0] - 90), detected_object.Left)
				new_right = max(min(self._camera_width - 1, detected_object.Center[0] + 90), detected_object.Right)
				cropped_left = self._left_frame[detected_object.Top:detected_object.Bottom, new_left:new_right]
				cropped_right = self._right_frame[detected_object.Top:detected_object.Bottom, new_left:new_right]
				# Calculate disparity map
				disparity = self.calculate_disparity(cropped_left, cropped_right)
				# required_point = (detected_object.Bottom - detected_object.Top)//2, (detected_object.Right - detected_object.Left)//2
				# Select the largest disparity (Closest point as required point) 
				required_point = np.unravel_index(disparity.argmax(), disparity.shape)
				# Calculate depth
				(x, y, z) = self.calculate_point_cloud(required_point, disparity[required_point])
				depth = np.sqrt(x**2 + y**2 + z**2)
				# Create bouding box
				cv2.rectangle(left_frame_copy, (new_left, detected_object.Top), (new_right, detected_object.Bottom), (255, 0, 0), 1) 
				# Overlay distance over objects
				left_frame_copy = cv2.putText(left_frame_copy, (detected_object.Class + ", Distance = " + str(np.round(depth, 2))), (new_left, detected_object.Top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

				detected_object.Distance = depth

			# Publish Object distance 
			try:        
				self._object_distance_frame_publisher.publish(self._bridge.cv2_to_imgmsg(left_frame_copy, encoding="rgb8"))
				self._object_distance_publisher.publish(detections_msg)
				rospy.loginfo("Published object distance")
			except CvBridgeError as e:
				print(e)

			self._processing = False


def main():
	rospy.init_node('stereovision_local')

	STEREO_MODE = rospy.get_param('stereo_mode', "StereoSGBM")
	CAMERA_BASELINE = rospy.get_param('baseline', 0.4)

	TOPICS_FROM = {'left_camera':"/prius/left_camera/image_raw",
			'right_camera':"/prius/right_camera/image_raw",
			'detections':"/object_detection/detected_objects"}

	TOPICS_TO = {'frame':"stereovision_local/image_object_distance", 'msg':"stereovision_local/object_distance"}
	
	LEFT_CAMERA_INFO = rospy.wait_for_message('/prius/right_camera/camera_info', CameraInfo)
	# right_camera_info = rospy.wait_for_message('/prius/right_camera/camera_info', CameraInfo)

	# Suppose both cameras are matched so send info of left camera only
	stereo_matcher = stereoMatcher(STEREO_MODE, TOPICS_TO, LEFT_CAMERA_INFO, CAMERA_BASELINE)

	left_camera_subscriber = message_filters.Subscriber(TOPICS_FROM['left_camera'], Image, buff_size=2**24, queue_size=2)
	right_camera_subscriber = message_filters.Subscriber(TOPICS_FROM['right_camera'], Image, buff_size=2**24, queue_size=2)
	detections_subscriber = rospy.Subscriber(TOPICS_FROM['detections'], detectionMsg, stereo_matcher.display_disparity, queue_size=1)

	camera_time_synchronizer = message_filters.ApproximateTimeSynchronizer([left_camera_subscriber, right_camera_subscriber], 2, 0.05)
	camera_time_synchronizer.registerCallback(stereo_matcher.read_frames)

	rate = rospy.Rate(20)

	try:
		while not rospy.is_shutdown(): 
			rate.sleep()
			rospy.spin()

	except rospy.ROSInterruptException:
		rospy.loginfo("Shutting down")

if __name__ == "__main__":
	main()
