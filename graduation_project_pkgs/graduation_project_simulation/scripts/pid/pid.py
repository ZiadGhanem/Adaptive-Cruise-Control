import time
import numpy as np
import rospy


class PID:
    def _init_(self,topic_to):
	self._t_previous = time.time()
	self._v_total_error = 0
	self._v_previous_error = 0
	self._t_previous = 0

	self._kp = 2
	self._ki = 1
	self._kd = 0.5

	self._pid_publisher = rospy.Publisher(topic_to, _int_created ,queue_size=1)


    def callback(self, velocity):
        t_current = time.time()

        dt = t_current - self._t_previous

        v_current_error = velocity.desired_velocity - velocity.current_velocity
        self._v_total_error += v_current_error * dt
        v_error_rate = (v_current_error - self._v_previous_error) / dt

        p_throttle = self._kp * v_current_error
        i_throttle = self._ki * self._v_total_error
        d_throttle = self._kd * v_error_rate

        longitudinal_output = p_throttle + i_throttle + d_throttle
        
	if longitudinal_output >= 0:
		throttle_output = np.fmax(np.fmin(longitudinal_output, 1.0), 0.0)
		brake_output = 0
	else:
		throttle_output = 0
		brake_output = np.fmax(np.fmin(longitudinal_output, 1.0), 0.0)

	try:
    		self._pid_publisher.publish(self._set_throttle)
    		rospy.loginfo("Published PID velocity")
	except Exceptionrospy.ROSInterruptException as e:
    		pass


        self._v_previous_error = v_current_error
        self._t_previous = t_current

def main():
	TOPIC_FROM = #
	TOPIC_TO = #
	rospy.init_node('pid')
	pid = PID(TOPIC_TO)
	controller_subscriber = rospy.Subscriber(TOPIC_FROM, _int_created ,pid.callback, queue_size = 1)

	try:
		while not rospy.is_shutdown():
			rospy.spin()
	except rospy.ROSInterruptException:
		rospy.loginfo("Shutting down")



if __name__ == "__main__":
	main()
