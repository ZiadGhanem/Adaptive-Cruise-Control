#!/usr/bin/env python3

import rospy
import cv2
import tensorflow as tf
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError 
import numpy as np
from lanenet_model import lanenet_merge_model
from config import global_config

class laneTracker:
	def __init__(self, camera_info, topic_to, weights_path):
		# Create publisher
		self._lanetracker_publisher = rospy.Publisher(topic_to, Image, queue_size = 1)
		self._bridge = CvBridge()

		self._CFG = global_config.cfg
		self._VGG_MEAN = [103.939, 116.779, 123.68]

		self._input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, self._CFG.TRAIN.IMG_HEIGHT, self._CFG.TRAIN.IMG_WIDTH, 3], name='input_tensor')
		phase_tensor = tf.constant('test', tf.string)
		net = lanenet_merge_model.LaneNet()
		self._binary_seg_ret, self._instance_seg_ret = net.test_inference(self._input_tensor, phase_tensor, 'lanenet_loss')
		initial_var = tf.compat.v1.global_variables()
		final_var = initial_var[:-1]
		saver = tf.compat.v1.train.Saver(final_var)
		# saver = tf.compat.v1.train.Saver()
		# Configure GPU usage
		sess_config = tf.compat.v1.ConfigProto(device_count={'GPU': 1})
		sess_config.gpu_options.per_process_gpu_memory_fraction = self._CFG.TEST.GPU_MEMORY_FRACTION
		sess_config.gpu_options.allow_growth = self._CFG.TRAIN.TF_ALLOW_GROWTH
		sess_config.gpu_options.allocator_type = 'BFC'
		# Create session and load weights
		self._sess = tf.compat.v1.Session(config=sess_config)
		saver.restore(sess=self._sess, save_path=weights_path)
		

	def process_img(self, img):
		img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
		img_resized = tf.image.resize(img, [self._CFG.TRAIN.IMG_HEIGHT, self._CFG.TRAIN.IMG_WIDTH], method=tf.image.ResizeMethod.BICUBIC)
		img_casted = tf.cast(img_resized, tf.float32)
		return tf.subtract(img_casted, self._VGG_MEAN)


	def pipeline(self, data):
		try:
			img = self._bridge.imgmsg_to_cv2(data, "rgb8")
		except CvBridgeError as e:
			print(e)


		img_processed = self.process_img(img)
		#img_processed = tf.map_fn(self.process_img, self._input_tensor, dtype=tf.float32)
		instance_seg_image, existence_output = self._sess.run([self._binary_seg_ret, self._instance_seg_ret],
								 feed_dict={self._input_tensor: img_processed})
		(instance_seg_image * 255).astype(np.uint8)
		print(instance_seg_image)
		print(instance_seg_image.shape)
		try:
			self._lanetracker_publisher.publish(self._bridge.cv2_to_imgmsg(instance_seg_image[0, :, :, 0], encoding="rgb8"))
		except CvBridgeError as e:
			print(e)

	def shutdown(self):
		self._sess.close()

def main():
	# Initialize node
	rospy.init_node("lane_tracking")

	# Get camera width and height
	LEFT_CAMERA_INFO = rospy.wait_for_message('/prius/right_camera/camera_info', CameraInfo)

	# Get model path
	WEIGHTS_PATH = rospy.get_param('weights_path', "/home/gradproj2020/catkin_ws/src/graduation_project_simulation/scripts/lanenet/model/culane_lanenet/culane_lanenet_vgg_2018-12-01-14-38-37.ckpt-10000")

	# Subscriber and publisher topics
	TOPIC_FROM = "/prius/left_camera/image_raw"
	TOPIC_TO = "lane_tracking/image_lane_tracking"

	lane_tracker = laneTracker(LEFT_CAMERA_INFO, TOPIC_TO, WEIGHTS_PATH)

	rospy.on_shutdown(lane_tracker.shutdown)


	# Left image subscriber
	rawframe_subscriber = rospy.Subscriber(TOPIC_FROM, Image, lane_tracker.pipeline, buff_size=2**24, queue_size=1)
	
		
	try:
		while not rospy.is_shutdown():
			rospy.spin()
	except rospy.ROSInterruptException:
		rospy.loginfo("Shutting down")

if __name__ == "__main__":
	main()

