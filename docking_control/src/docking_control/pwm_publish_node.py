#!/usr/bin/env python3

import rospy
from mavros_msgs.msg import OverrideRCIn


class PWMPublish:
    def __init__(self) -> None:
        self.control_pub = rospy.Publisher(
            "/mavros/rc/override", OverrideRCIn, queue_size=1
        )
        self.pwm_sub = rospy.Subscriber("/bluerov2_dock/pwm", OverrideRCIn, self.pwm_cb)
        self.pwm_data = None
        # self.pwm_data = OverrideRCIn()
        # self.pwm_data.channels = [OverrideRCIn.CHAN_NOCHANGE] * 18
        hz = 80
        self.pwm_timer = rospy.Timer(rospy.Duration(1 / hz), self.run)

    def pwm_cb(self, data):
        """Callback function for the PWM data.

        Args:
            data: PWM data
        """
        self.pwm_data = data

    def run(self, _):
        """Run the PWM publisher node.

        Args:
            _: Timer event
        """
        if self.pwm_data is not None:
            self.control_pub.publish(self.pwm_data)


if __name__ == "__main__":
    try:
        rospy.init_node("pwm_publisher", anonymous=True)
    except KeyboardInterrupt:
        rospy.logwarn("Shutting down the node")

    obj = PWMPublish()
    rospy.spin()
