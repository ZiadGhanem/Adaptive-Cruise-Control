#!/usr/bin/env python

import time
import numpy as np
import rospy
from prius_msgs.msg import Control
from sensor_msgs.msg import Joy
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry
from PyQt5 import QtWidgets, uic
import sys


class PID(QtWidgets.QWidget):
	def __init__(self,topics_from, topic_to):
		# Load UserInterface
		super(PID, self).__init__()
		uic.loadUi('/home/gradproj2020/catkin_ws/src/graduation_project_pkgs/graduation_project_simulation/scripts/pid/gui.ui', self)

		# Initial variables
		self._v_total_error = 0
		self._v_previous_error = 0
		self._t_previous = 0

		self._closest_distance = float('inf')
		self._current_velocity = 0
			
		self._circle_flag = True
		self._R1_flag = True
		self._R1_flag = True
		self._cruise_control_state = False
		self._cruise_control_speed = 0

		self._last_published_time = rospy.get_rostime()
		self._last_published = None


		# Contoller constants
		self._STEERING_AXIS = 0
		self._THROTTLE_AXIS = 5
		self._SQUARE = 0
		self._CROSS = 1
		self._CIRCLE = 2
		self._TRIANGLE = 3
		self._L1 = 4
		self._R1 = 5
		self._L2 = 6
		self._R2 = 7

		# PID constants
		self._KP = 0.2
		self._KI = 0.1
		self._KD = 0.05
		self._AMAX = 1.5
		self._MAX_SPEED = 80

		# Create subscribers
		self._controller_subscriber = rospy.Subscriber(topics_from['joystick_control'], Joy , self.joystick_callback, queue_size = 1)
		self._odometry_subscriber = rospy.Subscriber(topics_from['Odometry'], Odometry ,self.odometry_callback, queue_size = 1)
		self._closest_distance_subscriber = rospy.Subscriber(topics_from['closest_distance'], Float32 , self.closest_distance_callback, queue_size = 1)

		# Create Publishers
		self._prius_publisher = rospy.Publisher(topic_to, Control, queue_size=1)

		self._t_previous = time.time()

		self.timer = rospy.Timer(rospy.Duration(1./20.), self.timer_callback)

	def timer_callback(self, event):
		if self._last_published and self._last_published_time < rospy.get_rostime() + rospy.Duration(1.0/20.):
			self.joystick_callback(self._last_published)

	def odometry_callback(self, odometry):
		self._current_velocity = np.sqrt((odometry.twist.twist.linear.x)**2 + (odometry.twist.twist.linear.y)**2 + (odometry.twist.twist.linear.z)**2) * 3.6

	def closest_distance_callback(self, closest_distance):
		self._closest_distance = max(0, closest_distance.data - 1)

	def joystick_callback(self, message):
		command = Control()

		# Set message header
		command.header = message.header

		# Get steering value
		command.steer = message.axes[self._STEERING_AXIS]
		# Set GUI steering value
		self.steering_scrollbar.setValue(int(message.axes[self._STEERING_AXIS]* -100))

		# Get cruise control state
		if message.buttons[self._CIRCLE] and self._circle_flag:
			self._circle_flag = False
			self._cruise_control_state = not self._cruise_control_state
			self.cruise_radiobutton.setChecked(self._cruise_control_state)
		elif not message.buttons[self._CIRCLE]:
			self._circle_flag = True

		# Increment cruise control speed
		if message.buttons[self._R1] and self._R1_flag:
			self._R1_flag = False
			if self._cruise_control_speed < self._MAX_SPEED:
				self._cruise_control_speed += 1
				self.cruise_lcdnumber.display(self._cruise_control_speed)
		elif not message.buttons[self._R1]:
			self._R1_flag = True

		# Decrement cruise control speed
		if message.buttons[self._L1] and self._L1_flag:
			self._L1_flag = False
			if self._cruise_control_speed > 0:
				self._cruise_control_speed -= 1
				self.cruise_lcdnumber.display(self._cruise_control_speed)
		elif not message.buttons[self._L1]:
			self._L1_flag = True

		# If cruise control was on then 
		if self._cruise_control_state:
			command.shift_gears = Control.FORWARD
			self.gears_lineedit.setText('D')

			# Calculate the safe distance which prevents collision
			safe_velocity = np.sqrt(2 * self._AMAX * self._closest_distance) * 3.6
			# Get the minimum of the safe distance or the speed desired by the driver
			desired_velocity = min(self._cruise_control_speed, safe_velocity)

			# PID loop
			t_current = time.time()

			dt = t_current - self._t_previous

			v_current_error =  desired_velocity - self._current_velocity
			self._v_total_error += v_current_error * dt
			v_error_rate = (v_current_error - self._v_previous_error) / dt

			p_throttle = self._KP * v_current_error
			i_throttle = self._KI * self._v_total_error
			d_throttle = self._KD * v_error_rate

			longitudinal_output = p_throttle + i_throttle + d_throttle

			if longitudinal_output >= 0:
				command.throttle = np.fmax(np.fmin(longitudinal_output, 1.0), 0.0)
				command.brake = 0.0
				self.throttle_scrollbar.setValue(int(command.throttle * -100))
			else:
				command.throttle = 0.0
				command.brake = np.fmax(np.fmin(-longitudinal_output, 1.0), 0.0)
				self.throttle_scrollbar.setValue(int(command.brake * 100))


			self._v_previous_error = v_current_error
			self._t_previous = t_current
		else:
			# Reset variables
			self._v_total_error = 0
			self._v_previous_error = 0
			self._t_previous = 0

			self._closest_distance = float('inf')

			# Get throttle/breaking value
			if message.axes[self._THROTTLE_AXIS] >= 0:
				command.throttle = message.axes[self._THROTTLE_AXIS]
				command.brake = 0.0
			else:
				command.brake = message.axes[self._THROTTLE_AXIS] * -1
				command.throttle = 0.0

			# Set GUI throttle value
			self.throttle_scrollbar.setValue(int(message.axes[self._THROTTLE_AXIS] * -100))

			# Get gears value
			if message.buttons[self._TRIANGLE]:
				command.shift_gears = Control.FORWARD
				self.gears_lineedit.setText('D')
			elif message.buttons[self._CROSS]:
				command.shift_gears = Control.NEUTRAL
				self.gears_lineedit.setText('N')
			elif message.buttons[self._SQUARE]:
				command.shift_gears = Control.REVERSE
				self.gears_lineedit.setText('R')
			else:
				command.shift_gears = Control.NO_COMMAND

		self.velocity_lcdnumber.display(int(self._current_velocity))

		self._t_previous = time.time()

		# Send control message
		try:
			self._prius_publisher.publish(command)
			rospy.loginfo("Published commands")
		except Exceptionrospy.ROSInterruptException as e:
			pass

		self._last_published = message


def main():
	TOPICS_FROM = {'joystick_control':"joy",
			'Odometry':"base_pose_ground_truth",
			'closest_distance':"fake_lane_estimation/closest_object_distance"}
	TOPIC_TO = 'prius'

	rospy.init_node('pid')

	app = QtWidgets.QApplication(sys.argv)
	pid = PID(TOPICS_FROM, TOPIC_TO)
	pid.setWindowTitle('Control Window')
	pid.show()
	app.exec_()

	try:
		while not rospy.is_shutdown(): 
			rospy.spin()

	except rospy.ROSInterruptException:
		rospy.loginfo("Shutting down")

if __name__ == '__main__':
	main()
