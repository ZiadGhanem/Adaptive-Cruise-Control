#!/usr/bin/env python3


import cv2
import numpy as np
import rospy
import rospkg
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
    
class videoRecorder:
	def __init__(self, camera_info, frame_rate, output_path):
		# Read camera info
		self._camera_width = camera_info.width
		self._camera_height = camera_info.height

		# Create video writer
		self._video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'H264'), frame_rate, (self._camera_width, self._camera_height))

		# Create bridge for images
		self._bridge = CvBridge()


	def pipeline(self,data):
		try:
			img = self._bridge.imgmsg_to_cv2(data, "rgb8")
		except CvBridgeError as e:
			print(e)

		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		self._video_writer.write(img)
		cv2.imshow('Video Stream', img)
		if cv2.waitKey(10) & 0xFF == ord('q'):
			self._video_writer.release()
			cv2.destroyAllWindows()
			rospy.signal_shutdown("Shutting down")


def main():
	rospy.init_node('videorecorder', disable_signals=True)
	LEFT_CAMERA_INFO = rospy.wait_for_message('/prius/right_camera/camera_info', CameraInfo)
	TOPIC_FROM = "/prius/left_camera/image_raw"
	FRAME_RATE = 30
	OUTPUT_PATH = "/home/gradproj2020/test.avi"
	video_recorder = videoRecorder(LEFT_CAMERA_INFO, FRAME_RATE, OUTPUT_PATH)
	rawframe_subscriber = rospy.Subscriber(TOPIC_FROM, Image, video_recorder.pipeline, buff_size=2**24, queue_size=1)

	try:
		while not rospy.is_shutdown():
			rospy.spin()
	except rospy.ROSInterruptException:
		rospy.loginfo("Shutting down")


if __name__ == "__main__":
	main()

