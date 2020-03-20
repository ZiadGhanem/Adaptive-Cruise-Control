#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Uint32, Float32
from nav_msgs.msg import Odometry


class VelocityPlanner:
	def __init__(self,topic_to):
		self._Vmax = Vmax
		self._amax = amax
		self._amin = amin
		self._jpeak = jpeak		
		self._steps = 30
		self._cruise_control_state = cruise_control.CRUISE_CONTROL_OFF
		self._velocity_plan_publisher = rospy.Publisher(topicto, Uint32, queue_size=1)
		self._first_time = False

	def sigmoid_acc():
		tacc = np.linspace(0, 1, self._steps)
		velocity_plan = list()
		acceleration = current_acceleration
		for item in tacc:
			if (acceleration < self._amax):
				acceleration += self._jpeak
			else:
				acceleration = self._amax
		
			velocity_plan = np.append(velocity_plan, 1 / (1 + acceleration * np.exp(-item)))

		velocity_plan = np.interp(velocity_plan, (self._amin, self._amax), (self._current_velocity, np.min(self._desired_velocity, self._vmax)))
		return velocity_plan
		
	def sigmoid_dec(self):
		tdecc = np.linspace(0, 1, num=self._steps)
		velocity_plan = list()
		deceleration = current_acceleration
		for item in tdecc:
			if(deceleration > self._amin):
				deceleration = deceleration - self._jpeak
			else:
				deceleration = self._amin


		velocity_plan = np.append(velocity_plan, 1 / (1 + (-deceleration) * np.exp(-item)))
		velocity_plan = np.interp(velocity_plan, (self._amin, self._amax), (0, self._current_velocity))	
		return velocity_plan

		     
	def s_curve_motion(self):	
		safe_distance = -(self._current_velocity ** 2) / (2 * self._amin)

		if (self._closest_distance < safe_distance):
			velocity_plan = self.sigmoid_dec() 
		else:
			velocity_plan = self.sigmoid_acc()

		return velocity_plan
	
	def callback(self, odometry, closest_object_distance, cruise_control):
		self._current_velocity = np.sqrt((odometry.twist.twist.linear.x)**2 + (odometry.twist.twist.linear.y)**2 + (odometry.twist.twist.linear.z)**2) * 3.6
		self._cruise_control_state = cruise_control.cruise_control
		self._desired_velocity = cruise_control.cruise_control_speed
		self._closest_distance = closest_object_distance.data
		self._new_data = True

	def main_loop(self):
		rate = rospy.Rate(30)
		current_velocity_index = 0
		while not self._new_data:
			pass

		velocity_plan = self._s_curve_motion()

		try:
			while not rospy.is_shutdown():
				if self._cruise_control_state == cruise_control.CRUISE_CONTROL_ON:

					self._velocity_plan_publisher.publish(velocity_plan[current_velocity_index])

					if current_velocity_index < self._steps - 1:
						current_velocity_index += 1
					else:
						current_velocity_index = 0
						velocity_plan = self._s_curve_motion()

				else:
					current_velocity_index = 0
					velocity_plan = None

				rate.sleep()
				rospy.spin()



		except rospy.ROSInterruptException:
		rospy.loginfo("Shutting down")


def main():
	rospy.init_node('velocity_planner')
	TOPIC_TO = "velocity_planner/velocity_plan"
	TOPICS_FROM = {'odometry':"base_pose_ground_truth",
			'closest_distance':"fake_lane_estimation/closest_object_distance",
			'cruise_control':"cruise_control"}
	velocity_planner = VelocityPlanner(TOPIC_TO)

	odometry_subscriber = message_filters.Subscriber(TOPICS_FROM['odometry'], Odometry, buff_size=2**24, queue_size=1)
	closest_distance_subscriber = message_filters.Subscriber(TOPICS_FROM['closest_distance'], Float32, buff_size=2**24, queue_size=1)
	cruise_control_subscriber = message_filters.Subscriber(TOPICS_FROM['cruise_control'], Control, buff_size=2**24, queue_size=1)

	time_synchronizer = message_filters.ApproximateTimeSynchronizer([odometry_subscriber, closest_distance_subscriber, cruise_control_subscriber], 1, 0.05)
	time_synchronizer.registerCallback(velocity_planner.callback)

if __name__ == '__main__':
	try:
		main()
	except rospy.ROSInterruptException:
		pass
