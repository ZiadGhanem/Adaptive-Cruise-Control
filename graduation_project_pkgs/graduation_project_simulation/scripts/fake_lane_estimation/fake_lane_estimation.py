#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
import message_filters
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Float32
from sensor_msgs.msg import Image, CameraInfo
from graduation_project_simulation.msg import detectedObject, detectionMsg

class fakeLaneEstimator:
	def __init__(self, topics_to, camera_info):
		self._CAMERA_WIDTH = camera_info.width
		self._CAMERA_HEIGHT = camera_info.height

		self._LEFT_LINE_P1 = (0, self._CAMERA_HEIGHT - 1)
		self._LEFT_LINE_P2 = (int((self._CAMERA_WIDTH - 1) * 0.84), 0)

		self._LEFT_LINE_SLOPE = (self._LEFT_LINE_P2[1]-self._LEFT_LINE_P1[1])/(self._LEFT_LINE_P2[0]-self._LEFT_LINE_P1[0])
		self._LEFT_LINE_INTERCEPT = self._LEFT_LINE_P1[1] - (self._LEFT_LINE_SLOPE*self._LEFT_LINE_P1[0])

		self._RIGHT_LINE_P1 = ((self._CAMERA_WIDTH-1), (self._CAMERA_HEIGHT-1))
		self._RIGHT_LINE_P2 = (int((self._CAMERA_WIDTH-1) * 0.16), 0)

		self._RIGHT_LINE_SLOPE = (self._RIGHT_LINE_P2[1]-self._RIGHT_LINE_P1[1])/(self._RIGHT_LINE_P2[0]-self._RIGHT_LINE_P1[0])
		self._RIGHT_LINE_INTERCEPT = self._RIGHT_LINE_P1[1] - (self._RIGHT_LINE_SLOPE*self._RIGHT_LINE_P1[0])

		self._bridge = CvBridge()
		
		self._closest_distance_publisher = rospy.Publisher(topics_to['object'], Float32, queue_size=1)

		"""
		self._closest_object_frame_publisher = rospy.Publisher(topics_to['frame'], Image, queue_size=1)
		"""

	def right_line_check(self, point):
		if point[0] > ((point[1]-self._RIGHT_LINE_INTERCEPT)/self._RIGHT_LINE_SLOPE):
			return True
		else:
			return False

	def left_line_check(self, point):
		if point[0] < ((point[1]-self._LEFT_LINE_INTERCEPT)/self._LEFT_LINE_SLOPE):
			return True
		else:
			return False


	def callback(self, detections_msg):
		"""
		try:
			frame = self._bridge.imgmsg_to_cv2(left_frame, "rgb8")
		except CvBridgeError as e:
       			print(e)
		"""

		closest_distance = float('inf')
		closest_object = None

		for detected_object in detections_msg.detectedObjects:
			left_point = (detected_object.Left, detected_object.Bottom)
			right_point = (detected_object.Right, detected_object.Bottom)

			left_line_left_point_check = self.left_line_check(left_point)
			left_line_right_point_check = self.left_line_check(right_point)
			right_line_left_point_check = self.right_line_check(left_point)
			right_line_right_point_check = self.right_line_check(right_point)
			
			if(not((left_line_left_point_check and left_line_right_point_check) or (right_line_left_point_check and right_line_right_point_check))):
				if detected_object.Distance < closest_distance:
					closest_distance = detected_object.Distance
					closest_object = detected_object

		"""
		cv2.line(frame, self._LEFT_LINE_P1, self._LEFT_LINE_P2, (255, 255, 255), 4)
		cv2.line(frame, self._RIGHT_LINE_P1, self._RIGHT_LINE_P2, (255, 255, 255), 4)
		if closest_object is not None:
			# Create bouding box
				cv2.rectangle(frame, (closest_object.Left, closest_object.Top), (closest_object.Right, closest_object.Bottom), (255, 0, 0), 1) 
				# Overlay distance over objects
				frame = cv2.putText(frame, (closest_object.Class + ", Distance = " + str(np.round(closest_object.Distance, 2))), (closest_object.Left, closest_object.Top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
		"""

		try:    
			"""    
			self._closest_object_frame_publisher.publish(self._bridge.cv2_to_imgmsg(frame, encoding="rgb8"))
			"""
			self._closest_distance_publisher.publish(closest_distance)
			rospy.loginfo("Published object distance")
		except CvBridgeError as e:
			print(e)

def main():
	rospy.init_node('fake_lane_estimation')
	TOPICS_FROM = {'left_camera':"/prius/left_camera/image_raw", 'objects':"stereovision_local/object_distance"}
	TOPICS_TO = {'frame':"fake_lane_estimation/closest_object_frame", 'object':"fake_lane_estimation/closest_object_distance"}

	LEFT_CAMERA_INFO = rospy.wait_for_message('/prius/right_camera/camera_info', CameraInfo)

	fake_lane_estimator = fakeLaneEstimator(TOPICS_TO, LEFT_CAMERA_INFO)
	
	"""
	objects_subscriber = message_filters.Subscriber(TOPICS_FROM['objects'], detectionMsg, queue_size=1)
	left_camera_subscriber = message_filters.Subscriber(TOPICS_FROM['left_camera'], Image, buff_size=2**24, queue_size=2)

	synchronizer = message_filters.ApproximateTimeSynchronizer([objects_subscriber, left_camera_subscriber], 2, 0.05)
	synchronizer.registerCallback(fake_lane_estimator.callback)
	"""
	objects_subscriber = rospy.Subscriber(TOPICS_FROM['objects'], detectionMsg, fake_lane_estimator.callback, queue_size=1)

	rospy.spin()

if __name__ == "__main__":
	main()
