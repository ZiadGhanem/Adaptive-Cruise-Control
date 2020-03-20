#!/usr/bin/env python
import os
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

class stereoMatcher:
	def __init__(self, stereo_mode, topics_to, camera_matrix_file, translation_vector_file, camera_size):
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
		self._disparity_publisher = rospy.Publisher(topics_to['disparity'], CompressedImage, queue_size=1)
		# Create depth map publisher
		self._depthmap_publisher = rospy.Publisher(topics_to['depth'], CompressedImage, queue_size=1)


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

	def display_disparity(self, data):
		np_arr = np.fromstring(data.data, np.uint8)
		frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		# Separate left and right frames
		left_frame = frame[:, :self._FRAME_WIDTH]
		right_frame = frame[:, self._FRAME_WIDTH:]
		# Calculate disparity map
		disparity = self.calculate_disparity(left_frame, right_frame)
		# Normalize disparity map
		normalized_disparity = np.uint8(cv2.normalize(src=disparity, dst=None, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX))
		# Create colored disparity map
		colored_disparity = cv2.applyColorMap(normalized_disparity, cv2.COLORMAP_RAINBOW)
		# Publish disparity map
		try:        
			self._disparity_publisher.publish(self._bridge.cv2_to_compressed_imgmsg(colored_disparity))
			rospy.loginfo("Published disparity")
		except CvBridgeError as e:
			print(e)
		# Calculate depth map
		depth_map = self.calculate_depth(disparity)
		# Publish depth map
		try:        
			self._depthmap_publisher.publish(self._bridge.cv2_to_compressed_imgmsg(depth_map))
			rospy.loginfo("Published depth map")
		except CvBridgeError as e:
			print(e)




def main():
	rospy.init_node('stereovision')

	STEREO_MODE = rospy.get_param('stereo_mode', "StereoSGBM")
	CAMERA_MATRIX_FILE = rospy.get_param('camera_matrix_file', "/home/gradproj2020/catkin_ws/src/graduation_project/scripts/stereovision/Camera Matrix.npy")
	TRANSLATION_VECTOR_FILE = rospy.get_param('translation_vector_file', "/home/gradproj2020/catkin_ws/src/graduation_project/scripts/stereovision/Translation Vector.npy")
	CAMERA_WIDTH = rospy.get_param('camera_width', 1280)
	CAMERA_HEIGHT = rospy.get_param('camera_height', 480)

	TOPIC_FROM = "/calibrator/image_calibrated/compressed"
	TOPICS_TO = {'depth':"stereovision/depth_map", 'disparity':"stereovision/disparity/compressed"}

	stereo_matcher = stereoMatcher(STEREO_MODE, TOPICS_TO, CAMERA_MATRIX_FILE, TRANSLATION_VECTOR_FILE, (CAMERA_WIDTH, CAMERA_HEIGHT))
	stereocamera_subscriber = rospy.Subscriber(TOPIC_FROM, CompressedImage, stereo_matcher.display_disparity, buff_size=2**24, queue_size=1)
	rate = rospy.Rate(30)
	try:
		while not rospy.is_shutdown():
			rate.sleep()
			rospy.spin()

	except rospy.ROSInterruptException:
		rospy.loginfo("Shutting down")

if __name__ == "__main__":
	main()
