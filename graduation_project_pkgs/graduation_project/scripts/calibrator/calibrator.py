#!/usr/bin/env python

import cv2
import numpy as np
import rospy
import rospkg
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError

class stereoCamera:
	def __init__(self, topic_to, calibration_file, camera_size):
		calibration = np.load(calibration_file, allow_pickle=False)
		self._IMAGE_SIZE = tuple(calibration["imageSize"])
		self._LEFT_MAP_X = calibration["leftMapX"]
		self._LEFT_MAP_Y = calibration["leftMapY"]
		self._LEFT_ROI =  tuple(calibration["leftROI"])
		self._RIGHT_MAP_X = calibration["rightMapX"]
		self._RIGHT_MAP_Y = calibration["rightMapY"]
		self._RIGHT_ROI = tuple(calibration["rightROI"])

		# Camera frame width and height
		self._CAMERA_WIDTH = camera_size[0]
		self._CAMERA_HEIGHT = camera_size[1]

		# Each Frame width and height
		self._FRAME_WIDTH = self._CAMERA_WIDTH // 2
		self._FRAME_HEIGHT = self._CAMERA_HEIGHT
	
		self._bridge = CvBridge()
		# Create calibrated frame publisher
		self._calibratedframe_publisher = rospy.Publisher(topic_to, CompressedImage, queue_size=1)
		

	def calibrate_frame(self, data):
		# Decompress received frame
		np_arr = np.fromstring(data.data, np.uint8)
		frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		# Separate left and right frames
		left_frame = frame[:, :self._FRAME_WIDTH]
		right_frame = frame[:, self._FRAME_WIDTH:]
		# Calibrate left and right frames
		left_frame_calibrated = cv2.remap(left_frame, self._LEFT_MAP_X, self._LEFT_MAP_Y, cv2.INTER_LANCZOS4)
		right_frame_calibrated = cv2.remap(right_frame, self._RIGHT_MAP_X, self._RIGHT_MAP_Y, cv2.INTER_LANCZOS4)
		# Reconcatenate frames
		frame_full = cv2.hconcat([left_frame_calibrated, right_frame_calibrated])
		# Publish calibrated frame
		try:        
			self._calibratedframe_publisher.publish(self._bridge.cv2_to_compressed_imgmsg(frame_full))
			rospy.loginfo("Published Calibrated frame")
		except CvBridgeError as e:
			print(e)




def main():
	rospy.init_node('calibrator')
	
	CALIBRATION_FILE = rospy.get_param('calibration_file', "/home/gradproj2020/catkin_ws/src/graduation_project/scripts/calibrator/Calibration Matrix.npz")
	CAMERA_WIDTH = rospy.get_param('camera_width', 1280)
	CAMERA_HEIGHT = rospy.get_param('camera_height', 480)

	TOPIC_FROM = 'usb_cam/image_raw/compressed'
	TOPIC_TO = "calibrator/image_calibrated/compressed"

	stereo_camera = stereoCamera(TOPIC_TO, CALIBRATION_FILE, (CAMERA_WIDTH, CAMERA_HEIGHT))

	rawframe_subscriber = rospy.Subscriber(TOPIC_FROM, CompressedImage, stereo_camera.calibrate_frame, buff_size=2**24, queue_size=1)
	rate = rospy.Rate(30)
	try:
		while not rospy.is_shutdown():
			rate.sleep()
			rospy.spin()
	except rospy.ROSInterruptException:
		rospy.loginfo("Shutting down")


if __name__ == "__main__":
    main()
