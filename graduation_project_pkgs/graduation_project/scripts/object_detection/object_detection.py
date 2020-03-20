#!/usr/bin/env python
import jetson.inference
import jetson.utils
import rospy
import os
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from graduation_project.msg import detectedObject, detectionMsg

class objectDetector:
	def __init__(self, topics_to, network, labels_file):
		# Create labels name list
		with open(labels_file) as labels:
			lines = labels.readlines()
			self._object_class = [label.strip() for label in lines]
		# Camera frame width and height
		self._CAMERA_WIDTH = 1280
		self._CAMERA_HEIGHT = 480

		# Each Frame width and height
		self._FRAME_WIDTH = self._CAMERA_WIDTH // 2
		self._FRAME_HEIGHT = self._CAMERA_HEIGHT

		# Initialize network
		self._net = jetson.inference.detectNet(network, threshold=0.5)
	
		self._bridge = CvBridge()
		# Create object detection frame publisher
		self._objectdetection_publisher = rospy.Publisher(topics_to['frame'], CompressedImage, queue_size=1)
		# Detections publisher
		self._detections_publisher = rospy.Publisher(topics_to['detections'], detectionMsg, queue_size=1)

	
	def detect_object(self, data):
		# Decompress received frame
		np_arr = np.fromstring(data.data, np.uint8)
		frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		# Separate left frame
		left_frame = frame[:, :self._FRAME_WIDTH]
		rgba_frame = cv2.cvtColor(left_frame, cv2.COLOR_RGB2RGBA)
		# Convert image to CUDA
		cuda_mem = jetson.utils.cudaFromNumpy(rgba_frame)
		# Detect objects
		"""
		Avaialble attributes = ['Area', 'Bottom', 'Center', 'ClassID','Confidence', 'Contains', 'Height', 'Instance', 'Left', 'Right','Top', 'Width', '__class__', 			'__delattr__', '__doc__', '__format__', '__getattribute__', '__hash__', '__init__', '__new__', '__reduce__', '__reduce_ex__', '__repr__','__setattr__', '__sizeof__', 			'__str__', '__subclasshook__']

		"""
		detections = self._net.Detect(cuda_mem, self._FRAME_WIDTH, self._FRAME_HEIGHT)
		
		# Return frame to numpy arrray
		detection_frame = jetson.utils.cudaToNumpy(cuda_mem, self._FRAME_WIDTH, self._FRAME_HEIGHT, 4)
		detection_frame = cv2.cvtColor(detection_frame, cv2.COLOR_RGBA2RGB)
		# Publish detected objects frame
		try:        
			self._objectdetection_publisher.publish(self._bridge.cv2_to_compressed_imgmsg(detection_frame))
			rospy.loginfo("Published object detection frame")
		except CvBridgeError as e:
			print(e)
		# Copy detections into detection msg object
		detection_msg = detectionMsg()
		for detection in detections:
			detected_object = detectedObject()
			detected_object.Class = self._object_class[detection.ClassID]
			detected_object.Center = detection.Center
			detected_object.Left = detection.Left
			detected_object.Top = detection.Top
			detected_object.Right = detection.Right
			detected_object.Bottom = detection.Bottom
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
	LABELS_FILE = rospy.get_param('labels_file', '/home/gradproj2020/catkin_ws/src/graduation_project/scripts/object_detection/networks/SSD-Mobilenet-v2/ssd_coco_labels.txt')

	os.chdir('/home/gradproj2020/catkin_ws/src/graduation_project/scripts/object_detection')

	TOPIC_FROM = "/calibrator/image_calibrated/compressed"
	TOPICS_TO = {'frame':"/object_detection/image_detected_objects/compressed",
		     'detections':"/object_detection/detected_objects"}
	
	object_detector = objectDetector(TOPICS_TO, NETWORK, LABELS_FILE)

	rawframe_subscriber = rospy.Subscriber(TOPIC_FROM, CompressedImage, object_detector.detect_object, buff_size=2**24, queue_size=1)
	rate = rospy.Rate(30)
	try:
		while not rospy.is_shutdown():
			rate.sleep()
			rospy.spin()
			
	except rospy.ROSInterruptException:
		rospy.loginfo("Shutting down")



if __name__ == "__main__":
	main()
