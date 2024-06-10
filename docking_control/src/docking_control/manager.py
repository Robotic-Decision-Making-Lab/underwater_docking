#!/usr/bin/env python3

from copy import deepcopy
import rospy
from geographic_msgs.msg import GeoPoint, GeoPointStamped
from mavros_msgs.msg import OverrideRCIn, Param, ParamValue
from mavros_msgs.srv import CommandHome, MessageInterval
from mavros_msgs.srv import ParamPull, ParamGet, ParamSet, ParamSetRequest, ParamSetResponse
from std_srvs.srv import SetBool, SetBoolResponse


class Manager:
    """Provides an interface between custom controllers and the BlueROV2."""

    STOPPED_PWM = 1500

    def __init__(self):
        """Create a new control manager."""
        rospy.init_node("blue_manager", anonymous=True)

        self.num_thrusters = rospy.get_param("~num_thrusters", 8)
        self.timeout = rospy.get_param("~mode_change/timeout", 1.0)
        self.retries = rospy.get_param("~mode_change/retries", 3)

        self.passthrough_enabled = False

        self.thruster_params_backup = {f"SERVO{i}_FUNCTION": None for i in range(1, self.num_thrusters + 1)}

        # Publishers
        self.override_rc_in_pub = rospy.Publisher(
            "/mavros/rc/override", OverrideRCIn, queue_size=10
        )
        self.gp_origin_pub = rospy.Publisher(
            "/mavros/global_position/set_gp_origin", GeoPointStamped, queue_size=10
        )

        # Subscribers
        # self.param_event_sub = rospy.Subscriber(
        #     "/mavros/param/event", Param, self.backup_thruster_params_cb
        # )
        
        self.backup_thruster_params_cb()

        # Services
        self.set_pwm_passthrough_srv = rospy.Service(
            "/blue/cmd/enable_passthrough", SetBool, self.set_rc_passthrough_mode_cb
        )

        # Service clients
        self.set_param_srv_client = rospy.ServiceProxy(
            "/mavros/param/set", ParamSet
        )
        self.set_message_rates_client = rospy.ServiceProxy(
            "/mavros/set_message_interval", MessageInterval
        )
        self.set_home_pos_client = rospy.ServiceProxy(
            "/mavros/cmd/set_home", CommandHome
        )
        self.pull_param_client = rospy.ServiceProxy(
            "/mavros/param/pull", ParamPull)
        
        self.get_param_client = rospy.ServiceProxy(
            "/mavros/param/get", ParamGet)
        
        # Set the intervals at which the desired MAVLink messages are sent
        def set_message_rates(message_ids, message_rates):
            for msg, rate in zip(message_ids, message_rates):
                self.set_message_rates_client(
                    message_id=msg, message_rate=rate
                )

        message_request_rate = rospy.get_param("~message_intervals/request_interval", 30.0)
        message_ids = rospy.get_param("~message_intervals/ids", [31, 32])
        message_rates = rospy.get_param("~message_intervals/rates", [100.0, 100.0])

        # We set the intervals periodically to handle the situation in which
        # users launch QGC which requests different rates on boot.
        self.message_rate_timer = rospy.Timer(
            rospy.Duration(message_request_rate),
            lambda event: set_message_rates(message_ids, message_rates),
        )

        # Set the home position
        def set_home_pos(lat, lon, alt, yaw):
            self.set_home_pos_client(
                current_gps=False,
                yaw=yaw,
                latitude=lat,
                longitude=lon,
                altitude=alt,
            )

        # Set the home position. Some folks have discussed that this is necessary for
        # GUIDED mode
        home_lat = rospy.get_param("~home_position/latitude", 44.65870)
        home_lon = rospy.get_param("~home_position/longitude", -124.06556)
        home_alt = rospy.get_param("~home_position/altitude", 0.0)
        home_yaw = rospy.get_param("~home_position/yaw", 270.0)
        hp_request_rate = rospy.get_param("~home_position/request_interval", 30.0)

        # Similar to the message rates, we set the home position periodically to handle
        # the case in which the home position is set by QGC to a different location
        self.home_pos_timer = rospy.Timer(
            rospy.Duration(hp_request_rate),
            lambda event: set_home_pos(home_lat, home_lon, home_alt, home_yaw),
        )

        # Now, set the EKF origin. This is necessary to enable GUIDED mode and other
        # autonomy features with ArduSub
        origin_lat = rospy.get_param("~ekf_origin/latitude", 44.65870)
        origin_lon = rospy.get_param("~ekf_origin/longitude", -124.06556)
        origin_alt = rospy.get_param("~ekf_origin/altitude", 0.0)

        # Normally, we would like to set the QoS policy to use transient local
        # durability, but MAVROS uses the volatile durability setting for its
        # subscriber. Consequently, we need to publish this every once-in-a-while
        # to make sure that it gets set
        self.set_ekf_origin_timer = rospy.Timer(
            rospy.Duration(15.0),
            lambda event: self.set_ekf_origin_cb(
                GeoPoint(latitude=origin_lat, longitude=origin_lon, altitude=origin_alt)
            ),
        )

    @property
    def params_successfully_backed_up(self):
        """Whether or not the thruster parameters are backed up.

        Returns:
            Whether or not the parameters are backed up.
        """
        return None not in self.thruster_params_backup.values()

    # def backup_thruster_params_cb(self, event):
    #     """Backup the default thruster parameter values.

    #     MAVROS publishes the parameter values of all ArduSub parameters when
    #     populating the parameter server. We subscribe to this topic to receive the
    #     messages at the same time as MAVROS so that we can avoid duplicating requests
    #     to the FCU.

    #     Args:
    #         event: The default parameter loading event triggered by MAVROS.
    #     """
    #     if (
    #         event.param_id in self.thruster_params_backup
    #         and self.thruster_params_backup[event.param_id] is None
    #     ):
    #         self.thruster_params_backup[event.param_id] = ParamValue(
    #             real=event.value
    #         )

    #         if self.params_successfully_backed_up:
    #             rospy.loginfo("Successfully backed up the thruster parameters.")
                
    def backup_thruster_params_cb(self):
        """Backup the default thruster parameter values.

        MAVROS publishes the parameter values of all ArduSub parameters when
        populating the parameter server. We subscribe to this topic to receive the
        messages at the same time as MAVROS so that we can avoid duplicating requests
        to the FCU.

        Args:
            event: The default parameter loading event triggered by MAVROS.
        """
        event_ids = self.thruster_params_backup.keys()
        j = 33
        for i in event_ids:
            self.thruster_params_backup[i] = ParamValue(real=j)
            j += 1
        
        if self.params_successfully_backed_up:
            rospy.loginfo("Successfully backed up the thruster parameters.")

    def set_rc(self, pwm):
        """Set the PWM values of the thrusters.

        Args:
            pwm: The PWM values to set the thruster to. The length of the provided list
                must be equal to the number of thrusters.
        """
        if len(pwm) != self.num_thrusters:
            raise ValueError(
                "The length of the PWM input must equal the number of thrusters."
            )

        # Change the values of only the thruster channels
        channels = pwm + [OverrideRCIn.CHAN_NOCHANGE] * (18 - self.num_thrusters)
        self.override_rc_in_pub.publish(OverrideRCIn(channels=channels))

    def stop_thrusters(self):
        """Stop all thrusters."""
        rospy.logwarn("Stopping all BlueROV2 thrusters.")
        self.set_rc([self.STOPPED_PWM] * self.num_thrusters)

    def set_thruster_params(self, params):
        """Set the thruster parameters.

        Args:
            params: The ArduSub parameters to set.

        Returns:
            True if the parameters were successfully set, False otherwise.
        """
        responses = []
        for key, value in params.items():
            request = ParamSetRequest()
            request.param_id = key 
            request.value = value
            response = self.set_param_srv_client(request)
            responses.append(response.success)
        return all(responses)

    def set_rc_passthrough_mode_cb(self, request):
        """Set the RC Passthrough mode.

        RC Passthrough mode enables users to control the BlueROV2 thrusters directly
        using the RC channels. It is important that users disable their RC transmitter
        prior to enabling RC Passthrough mode to avoid sending conflicting commands to
        the thrusters.

        Args:
            request: The request to enable/disable RC passthrough mode.

        Returns:
            The result of the request.
        """
        response = SetBoolResponse()

        if request.data:
            if self.passthrough_enabled:
                response.success = True
                response.message = "The system is already in RC Passthrough mode."
                return response

            if not self.thruster_params_backup:
                response.success = False
                response.message = (
                    "The thrusters cannot be set to RC Passthrough mode without first"
                    " being successfully backed up."
                )
                return response

            rospy.logwarn(
                "Attempting to switch to the RC Passthrough flight mode. All ArduSub"
                " arming and failsafe procedures will be disabled upon success."
            )

            passthrough_params = deepcopy(self.thruster_params_backup)

            # Set the servo mode to "RC Passthrough"
            # This disables the arming and failsafe features, but now lets us send PWM
            # values to the thrusters without any mixing
            try:
                for param in passthrough_params.values():
                    param.real = 1.0
            except AttributeError:
                response.success = False
                response.message = (
                    "Failed to switch to RC Passthrough mode. Please ensure that all"
                    " ArduSub parameters have been loaded prior to attempting to"
                    " switch modes."
                )
                return response

            for _ in range(self.retries):
                self.passthrough_enabled = self.set_thruster_params(passthrough_params)
                response.success = self.passthrough_enabled

                if response.success:
                    break

            if response.success:
                response.message = "Successfully switched to RC Passthrough mode."

                self.stop_thrusters()
            else:
                response.message = "Failed to switch to RC Passthrough mode."
        else:
            if not self.thruster_params_backup:
                response.success = False
                response.message = (
                    "The thruster backup parameters have not yet been stored."
                )

            if not self.passthrough_enabled:
                response.success = True
                response.message = (
                    "The system was not in the RC Passthrough mode to start with."
                )
                return response

            self.stop_thrusters()

            rospy.logwarn("Attempting to disable RC Passthrough mode.")

            for _ in range(self.retries):
                self.passthrough_enabled = not self.set_thruster_params(self.thruster_params_backup)

                response.success = not self.passthrough_enabled

                if response.success:
                    break

            if response.success:
                response.message = "Successfully left RC Passthrough mode."
            else:
                rospy.logwarn(
                    "Failed to leave the RC Passthrough mode. If failure persists,"
                    " the following backup parameters may be restored manually using"
                    f" QGC: {self.thruster_params_backup}"
                )
                response.message = (
                    "Failed to disable RC Passthrough mode. Good luck soldier."
                )

        rospy.loginfo(response.message)

        return response

    def set_ekf_origin_cb(self, origin):
        """Set the EKF origin.

        This is required for navigation on a vehicle with one of the provided
        localizers.

        Args:
            origin: The EKF origin to set.
        """
        origin_stamped = GeoPointStamped()
        origin_stamped.header.stamp = rospy.Time.now()
        origin_stamped.position = origin
        self.gp_origin_pub.publish(origin_stamped)


def main():
    """Run the ROV manager."""
    manager = Manager()
    rospy.spin()


if __name__ == "__main__":
    main()
