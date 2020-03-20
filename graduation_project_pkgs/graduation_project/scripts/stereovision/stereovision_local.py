#!/usr/bin/env python3

import rospy
import os
import numpy as np
import cv2
import message_filters
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from graduation_project.msg import detectedObject, detectionMsg


class stereoMatcher:
	def __init__(self, stereo_mode, topic_to, camera_matrix_file, translation_vector_file, camera_size):
		# Load Camera matrix, Translation vector
		camera_matrix = np.load(camera_matrix_file)
		translation_vector = np.load(translation_vector_file)
		# Read focal length
		self._FOCAL_LENGTH = camera_matrix[0, 0]
		# Read baseline distance
		self._STEREO_BASELINE = - translation_vector[0] * 2.54

		# Camera frame width and height
		self._CAMERA_WIDTH = camera_size[0]
		self._CAMERA_HEIGHT = camera_size[1]

		# Each Frame width and height
		self._FRAME_WIDTH = self._CAMERA_WIDTH // 2
		self._FRAME_HEIGHT = self._CAMERA_HEIGHT

		self._BLOCK_SIZE = 5
		self._MIN_DISPARITY = 0
		self._NUM_DISPARITIES = 160
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
		self._object_distance_frame_publisher = rospy.Publisher(topic_to, CompressedImage, queue_size=1)


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

	def calculate_depth(self, disparity):
		return self._FOCAL_LENGTH * self._STEREO_BASELINE / disparity

	def display_disparity(self, calibrated_image, object_detection_image, detection_msg):
		rospy.loginfo("Received data")

		# Read calibrated frame
		np_arr = np.fromstring(calibrated_image.data, np.uint8)
		calibrated_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

		# Read calibrated frame
		np_arr = np.fromstring(object_detection_image.data, np.uint8)
		objectdetection_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

		# Separate left and right frames of calibrated image
		left_frame = calibrated_frame[:, :self._FRAME_WIDTH]
		right_frame = calibrated_frame[:, self._FRAME_WIDTH:]
		# Measure distance to objects in calibrated image
		for detected_object in detection_msg.detectedObjects:
			cropped_left = left_frame[detected_object.Top:detected_object.Bottom, detected_object.Left:detected_object.Right]
			cropped_right = right_frame[detected_object.Top:detected_object.Bottom, detected_object.Left:detected_object.Right]
			# Detect contours
			objects_threshold = cv2.adaptiveThreshold(cv2.cvtColor(cropped_left, cv2.COLOR_RGB2GRAY), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,2)
			objects_contour, _ = cv2.findContours(objects_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
			if len(objects_contour) > 0:
				# Get biggest contour
				biggest_contour = max(objects_contour, key=cv2.contourArea)
				contour_moment = cv2.moments(biggest_contour)
				# Get center of biggest contour
				if contour_moment["m00"] > 0:
					contour_center_x = int(contour_moment["m10"] / contour_moment["m00"])
					contour_center_y = int(contour_moment["m01"] / contour_moment["m00"])
					# Calculate disparity map
					disparity = self.calculate_disparity(cropped_left, cropped_right)
					# Calculate depth
					depth = self.calculate_depth(disparity[contour_center_y, contour_center_x])
					# Overlay distance over objects
					if 0 < depth < 300:
						objectdetection_frame = cv2.putText(objectdetection_frame, str(np.round(depth[0], 2)), detected_object.Center, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
				biggest_contour[:, :, 0] += detected_object.Left
				biggest_contour[:, :, 1] += detected_object.Top
				cv2.drawContours(objectdetection_frame, biggest_contour, -1, (0, 255, 0), 3) 
		# Publish Object distance 
		try:        
			self._object_distance_frame_publisher.publish(self._bridge.cv2_to_compressed_imgmsg(objectdetection_frame))
			rospy.loginfo("Published object distance frame")
		except CvBridgeError as e:
			print(e)



def main():
	rospy.init_node('stereovision_local')

	STEREO_MODE = rospy.get_param('stereo_mode', "StereoSGBM")
	CAMERA_MATRIX_FILE = rospy.get_param('camera_matrix_file', "/home/gradproj2020/catkin_ws/src/graduation_project/scripts/stereovision/Camera Matrix.npy")
	TRANSLATION_VECTOR_FILE = rospy.get_param('translation_vector_file', "/home/gradproj2020/catkin_ws/src/graduation_project/scripts/stereovision/Translation Vector.npy")
	CAMERA_WIDTH = rospy.get_param('camera_width', 1280)
	CAMERA_HEIGHT = rospy.get_param('camera_height', 480)

	TOPICS_FROM = {'calibrated_frame':"/calibrator/image_calibrated/compressed",
			'objectdetection_frame':"/object_detection/image_detected_objects/compressed",
		       'detections':"/object_detection/detected_objects"}
	TOPIC_TO = "stereovision_local/image_object_distance/compressed"

	CAMERA_MATRIX_FILE = "/home/gradproj2020/catkin_ws/src/graduation_project/scripts/stereovision/Camera Matrix.npy"
	TRANSLATION_VECTOR_FILE = "/home/gradproj2020/catkin_ws/src/graduation_project/scripts/stereovision/Translation Vector.npy"

	
	stereo_matcher = stereoMatcher(STEREO_MODE, TOPIC_TO, CAMERA_MATRIX_FILE, TRANSLATION_VECTOR_FILE, (CAMERA_WIDTH, CAMERA_HEIGHT))

	stereocamera_subscriber = message_filters.Subscriber(TOPICS_FROM['calibrated_frame'], CompressedImage, buff_size=2**24, queue_size=1)
	objectdetection_subscriber = message_filters.Subscriber(TOPICS_FROM['objectdetection_frame'], CompressedImage, buff_size=2**24, queue_size=1)
	detections_subscriber = message_filters.Subscriber(TOPICS_FROM['detections'], detectionMsg)

	time_synchronizer = message_filters.ApproximateTimeSynchronizer([stereocamera_subscriber, objectdetection_subscriber, detections_subscriber], 1, 0.05, allow_headerless=True)
	time_synchronizer.registerCallback(stereo_matcher.display_disparity)
	rate = rospy.Rate(30)
	try:
		while not rospy.is_shutdown(): 
			rate.sleep()
			rospy.spin()

	except rospy.ROSInterruptException:
		rospy.loginfo("Shutting down")

if __name__ == "__main__":
	main()
