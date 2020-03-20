#!/usr/bin/env python3
import jetson.inference
import jetson.utils
import rospy
import os
import numpy as np
import cv2
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
from graduation_project_simulation.msg import detectedObject, detectionMsg

class objectDetector:
	def __init__(self, topic_to, network, labels_file, camera_info):
		# Create labels name list
		with open(labels_file) as labels:
			lines = labels.readlines()
			self._object_class = [label.strip() for label in lines]

		# Camera width and height
		self._CAMERA_WIDTH = camera_info.width
		self._CAMERA_HEIGHT = camera_info.height

		# Initialize network
		self._net = jetson.inference.detectNet(network, threshold=0.5)
	
		self._bridge = CvBridge()

		# Detections publisher
		self._detections_publisher = rospy.Publisher(topic_to, detectionMsg, queue_size=1)
		
		# Desired classes
		self._desired_classes = ("person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light", "fire hydrant", "street sign", "stop sign",
					"parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "chair")

	
	def detect_object(self, data):
		try:
			frame = self._bridge.imgmsg_to_cv2(data, "rgb8")
		except CvBridgeError as e:
       			print(e)
		# Convert frame to RGBA
		rgba_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
		# Convert frame to CUDA
		cuda_mem = jetson.utils.cudaFromNumpy(rgba_frame)
		# Detect objects
		"""
		Avaialble attributes = ['Area', 'Bottom', 'Center', 'ClassID','Confidence', 'Contains', 'Height', 'Instance', 'Left', 'Right','Top', 'Width', '__class__', 			'__delattr__', '__doc__', '__format__', '__getattribute__', '__hash__', '__init__', '__new__', '__reduce__', '__reduce_ex__', '__repr__','__setattr__', '__sizeof__', 			'__str__', '__subclasshook__']

		"""
		detections = self._net.Detect(cuda_mem, self._CAMERA_WIDTH, self._CAMERA_HEIGHT)
		
		"""
		# Return frame to numpy arrray
		detection_frame = jetson.utils.cudaToNumpy(cuda_mem, self._CAMERA_WIDTH, self._CAMERA_HEIGHT, 4)
		detection_frame = cv2.cvtColor(detection_frame, cv2.COLOR_RGBA2RGB)
		detection_frame = detection_frame.astype(np.uint8)
		# Publish detected objects frame
		try:        
			self._objectdetection_publisher.publish(self._bridge.cv2_to_imgmsg(detection_frame, encoding="rgb8"))
			rospy.loginfo("Published object detection frame")
		except CvBridgeError as e:
			print(e)
		"""

		# Copy detections into detection msg object
		detection_msg = detectionMsg()
		detection_msg.header = data.header
		for detection in detections:
			if self._object_class[detection.ClassID] in self._desired_classes:
				detected_object = detectedObject()
				detected_object.Class = self._object_class[detection.ClassID]
				detected_object.Center.append(int(detection.Center[0]))
				detected_object.Center.append(int(detection.Center[1]))
				detected_object.Left = int(detection.Left)
				detected_object.Top = int(detection.Top)
				detected_object.Right = int(detection.Right)
				detected_object.Bottom = int(detection.Bottom)
				detection_msg.detectedObjects.append(detected_object)
		# Publish detections msg
		try:
			self._detections_publisher.publish(detection_msg)
			rospy.loginfo("Published object detection data")
		except CvBridgeError as e:
			print(e)


def main():
	rospy.init_node('object_detection')
	NETWORK = rospy.get_param('network', "ssd-mobilenet-v2")
	LABELS_FILE = rospy.get_param('labels_file', '/home/gradproj2020/catkin_ws/src/graduation_project_pkgs/graduation_project_simulation/scripts/object_detection/networks/SSD-Mobilenet-v2/ssd_coco_labels.txt')

	LEFT_CAMERA_INFO = rospy.wait_for_message('/prius/right_camera/camera_info', CameraInfo)

	os.chdir('/home/gradproj2020/catkin_ws/src/graduation_project_pkgs/graduation_project_simulation/scripts/object_detection')

	TOPIC_FROM = "/prius/left_camera/image_raw"
	TOPIC_TO = "/object_detection/detected_objects"
	
	object_detector = objectDetector(TOPIC_TO, NETWORK, LABELS_FILE, LEFT_CAMERA_INFO)

	rawframe_subscriber = rospy.Subscriber(TOPIC_FROM, Image, object_detector.detect_object, buff_size=2**24, queue_size=2)
	rate = rospy.Rate(30)
	try:
		while not rospy.is_shutdown():
			rate.sleep()
			rospy.spin()
			
	except rospy.ROSInterruptException:
		rospy.loginfo("Shutting down")



if __name__ == "__main__":
	main()
