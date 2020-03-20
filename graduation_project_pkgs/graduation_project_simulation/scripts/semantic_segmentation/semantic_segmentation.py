#!/usr/bin/env python3
import jetson.inference
import jetson.utils
import rospy
import os
import numpy as np
import cv2
import ctypes
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError


class semanticSegmentation:
	def __init__(self, topics_to, network, labels_file, camera_info):
		# Create labels name list
		with open(labels_file) as labels:
			lines = labels.readlines()
			self._object_class = [label.strip() for label in lines]

		# Camera width and height
		self._CAMERA_WIDTH = camera_info.width
		self._CAMERA_HEIGHT = camera_info.height
	
		self._FILTER_MODE = "point"
		self._IGNORE_CLASS = "void"
		self._ALPHA = 175.0

		# Initialize network
		self._net = jetson.inference.segNet(network)
		# set the alpha blending value
		self._net.SetOverlayAlpha(self._ALPHA)
		# allocate the output images for the overlay & mask
		self._img_overlay = jetson.utils.cudaAllocMapped(self._CAMERA_WIDTH * self._CAMERA_HEIGHT * 4 * ctypes.sizeof(ctypes.c_float))
		self._img_mask = jetson.utils.cudaAllocMapped(self._CAMERA_WIDTH * self._CAMERA_HEIGHT * 4 * ctypes.sizeof(ctypes.c_float))
	
		self._bridge = CvBridge()
		# Create semantic segmentation overlay and mask frame publisher
		self._overlay_publisher = rospy.Publisher(topics_to['overlay'], Image, queue_size=1)
		self._mask_publisher = rospy.Publisher(topics_to['mask'], Image, queue_size=1)

	def detect(self, data):
		# Receive frame from camera
		try:
			frame = self._bridge.imgmsg_to_cv2(data, "rgb8")
		except CvBridgeError as e:
       			print(e)

		# Convert frame to RGBA
		rgba_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
		# Convert frame to CUDA
		cuda_mem = jetson.utils.cudaFromNumpy(rgba_frame)

		# process the segmentation network
		self._net.Process(cuda_mem, self._CAMERA_WIDTH, self._CAMERA_HEIGHT, self._IGNORE_CLASS)

		# generate the overlay and mask
		self._net.Overlay(self._img_overlay, self._CAMERA_WIDTH, self._CAMERA_HEIGHT, self._FILTER_MODE)
		self._net.Mask(self._img_mask, self._CAMERA_WIDTH, self._CAMERA_HEIGHT, self._FILTER_MODE)

		# Return frame to numpy arrray
		overlay_frame = jetson.utils.cudaToNumpy(self._img_overlay, self._CAMERA_WIDTH, self._CAMERA_HEIGHT, 4)
		mask_frame = jetson.utils.cudaToNumpy(self._img_mask, self._CAMERA_WIDTH, self._CAMERA_HEIGHT, 4)
		
		# Convert RGBA frame to RGB
		overlay_frame = cv2.cvtColor(overlay_frame, cv2.COLOR_RGBA2RGB).astype(np.uint8)
		mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_RGBA2RGB).astype(np.uint8)
		
		test_frame = np.zeros_like(mask_frame)
		test_frame[np.where((mask_frame==[220, 20, 60]).all(axis=2))] = (255, 255, 255)

		# Publish semantic segmentation frame
		try:        
			self._overlay_publisher.publish(self._bridge.cv2_to_imgmsg(overlay_frame, encoding="rgb8"))
			self._mask_publisher.publish(self._bridge.cv2_to_imgmsg(test_frame, encoding="rgb8"))
			rospy.loginfo("Published semantic segmentation frame")
		except CvBridgeError as e:
			print(e)



def main():
	rospy.init_node('semantic_segmentation')
	NETWORK = rospy.get_param('network', "fcn-resnet18-cityscapes-512x256")
	LABELS_FILE = rospy.get_param('labels_file', '/home/gradproj2020/catkin_ws/src/graduation_project_simulation/scripts/semantic_segmentation/networks/FCN-ResNet18-Cityscapes-512x256/classes.txt')

	LEFT_CAMERA_INFO = rospy.wait_for_message('/prius/right_camera/camera_info', CameraInfo)

	os.chdir('/home/gradproj2020/catkin_ws/src/graduation_project_simulation/scripts/semantic_segmentation')

	TOPIC_FROM = "/prius/left_camera/image_raw"
	TOPICS_TO = {'overlay':"/semantic_segmentation/image_overlay",
		     'mask':"/semantic_segmentation/image_mask"}
	
	semantic_segmentation = semanticSegmentation(TOPICS_TO, NETWORK, LABELS_FILE, LEFT_CAMERA_INFO)

	rawframe_subscriber = rospy.Subscriber(TOPIC_FROM, Image, semantic_segmentation.detect, buff_size=2**24, queue_size=2)
	rate = rospy.Rate(30)
	try:
		while not rospy.is_shutdown():
			rate.sleep()
			rospy.spin()
			
	except rospy.ROSInterruptException:
		rospy.loginfo("Shutting down")



if __name__ == "__main__":
	main()
