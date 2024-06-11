#!/usr/bin/env python3

import rospy
import time
import numpy as np
import pandas as pd
import os
from auto_dock import MPControl
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from geometry_msgs.msg import PoseStamped, WrenchStamped
from sensor_msgs.msg import Joy, BatteryState, FluidPressure
from nav_msgs.msg import Odometry
from mavros_msgs.msg import OverrideRCIn, State
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from mavros_msgs.srv import CommandBool
from std_srvs.srv import SetBool
from scipy.spatial.transform import Rotation as R


class BlueROV2:
    def __init__(self) -> None:
        # Set flag for manual or auto mode
        self.mode_flag = "manual"

        # Set boolean to note when to send a blank button
        # (needed to get lights to turn on/off)
        self.reset_button_press = False

        # Set up pulse width modulation (pwm) values
        self.neutral_pwm = 1500
        # self.max_pwm_auto = 1600
        # self.min_pwm_auto = 1400
        self.max_pwm_manual = 1700
        self.min_pwm_manual = 1300
        self.max_possible_pwm = 1900
        self.min_possible_pwm = 1100

        self.deadzone_pwm = [1470, 1530]

        # Provide access to TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)

        self.override = None

        self.mocap_flag = False

        self.rov_odom = None
        self.dock_odom = None

        self.first_pose_flag = True
        self.previous_rov_pose = None
        self.rov_pose_sub_time = None

        self.first_fid_flag = True
        self.previous_fid_pose = None
        self.fid_pose_sub_time = None
        self.rov_pose = None
        self.rov_twist = None

        self.image_idx = 0
        self.light_level = 1500
        self.rc_passthrough_flag = False

        self.load_pwm_lookup()

        self.mpc = MPControl()

        # Set up dictionary to store subscriber data
        self.sub_data_dict = {}

        self.initialize_subscribers()
        self.initialize_publishers()
        self.initialize_services()
        # self.initialize_timers()

    def load_pwm_lookup(self):
        """Load the lookup table for converting thrust to pwm values"""
        cwd = os.path.dirname(__file__)
        csv = pd.read_csv(cwd + "/../../data/T200_data_16V.csv")

        thrust_vals = csv["Force"].tolist()
        neg_thrust = [i for i in thrust_vals if i < 0]
        pos_thrust = [i for i in thrust_vals if i > 0]
        zero_thrust = [i for i in thrust_vals if i == 0]

        pwm_vals = csv["PWM"].tolist()
        neg_t_pwm = [pwm_vals[i] for i in range(len(neg_thrust))]
        [
            pwm_vals[i]
            for i in range(len(neg_thrust), len(neg_thrust) + len(zero_thrust))
        ]
        pos_t_pwm = [
            pwm_vals[i]
            for i in range(len(neg_thrust) + len(zero_thrust), len(thrust_vals))
        ]

        self.neg_thrusts = np.array(neg_thrust)
        self.pos_thrusts = np.array(pos_thrust)
        self.neg_pwm = np.array(neg_t_pwm)
        self.pos_pwm = np.array(pos_t_pwm)

    def initialize_subscribers(self):
        # Set up subscribers
        self.joy_sub = rospy.Subscriber("/joy", Joy, self.store_sub_data, "joy")
        self.battery_sub = rospy.Subscriber(
            "/mavros/battery", BatteryState, self.store_sub_data, "battery"
        )
        self.state_subs = rospy.Subscriber(
            "/mavros/state", State, self.store_sub_data, "state"
        )
        self.pressure_sub = rospy.Subscriber(
            "/mavros/imu/static_pressure", FluidPressure, self.pressure_cb
        )
        self.rov_pose_sub = rospy.Subscriber(
            "/docking_control/vision_pose/pose", PoseStamped, self.rov_pose_cb
        )
        # self.rov_vel_sub = rospy.Subscriber(
        #     "/mavros/local_position/velocity_body", TwistStamped, self.rov_vel_cb
        # )

    def initialize_publishers(self):
        # Set up publishers
        # self.control_pub = rospy.Publisher(
        #     "/mavros/rc/override", OverrideRCIn, queue_size=1
        # )
        self.control_pub = rospy.Publisher(
            "/docking_control/pwm", OverrideRCIn, queue_size=1
        )
        self.mpc_pwm_pub = rospy.Publisher(
            "/docking_control/mpc_pwm", OverrideRCIn, queue_size=1
        )
        self.mpc_output = rospy.Publisher(
            "/docking_control/mpc", Float32MultiArray, queue_size=1
        )
        self.rov_odom_pub = rospy.Publisher(
            "/docking_control/rov_odom", Odometry, queue_size=1
        )
        self.mpc_wrench_pub = rospy.Publisher(
            "/docking_control/mpc_wrench", WrenchStamped, queue_size=1
        )

    def initialize_services(self):
        # Initialize arm/disarm service
        self.arm_srv = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        # Enable/Disable RC Passthrough Mode
        self.rc_passthrough_srv = rospy.ServiceProxy(
            "/blue/cmd/enable_passthrough", SetBool
        )

    def initialize_timers(self):
        """Initialize the timer for the control loop"""
        rospy.Timer(rospy.Duration(0.05), self.timer_cb)

    def wrap_pi2negpi(self, angle):
        """Wrap angle to [-pi, pi]"""
        return ((angle + np.pi) % (2 * np.pi)) - np.pi

    def pressure_cb(self, data):
        """Callback function for the pressure sensor

        Args:
            data: FluidPressure message
        """
        try:
            pressure = data.fluid_pressure
            rho = 1000
            g = 9.8
            self.depth = pressure / (rho * g)
        except Exception:
            rospy.logerr_throttle(
                10, "[BlueROV2][pressure_cb] Not receiving pressure readings"
            )

    def rov_pose_cb(self, pose):
        """Callback function for the ROV's pose

        Args:
            pose: PoseStamped message
        """

        try:
            x = pose.pose.orientation.x
            y = pose.pose.orientation.y
            z = pose.pose.orientation.z
            w = pose.pose.orientation.w
            euler = R.from_quat([x, y, z, w]).as_euler("xyz")
            euler[0]
            -euler[1]
            yaw = -euler[2]
            self.rov_pose = np.zeros((6, 1))
            self.rov_pose[0][0] = pose.pose.position.x
            self.rov_pose[1][0] = -pose.pose.position.y
            self.rov_pose[2][0] = -pose.pose.position.z
            self.rov_pose[3][0] = 0.0
            self.rov_pose[4][0] = 0.0
            self.rov_pose[5][0] = yaw

            if self.first_pose_flag:
                self.rov_pose_sub_time = pose.header.stamp.to_sec()
                self.previous_rov_pose = self.rov_pose
                self.first_pose_flag = False
            else:
                del_time = pose.header.stamp.to_sec() - self.rov_pose_sub_time
                rov_pose_diff = self.rov_pose - self.previous_rov_pose
                rov_twist = rov_pose_diff / del_time

                self.rov_twist = rov_twist

                self.rov_pose_sub_time = time.time()
                self.previous_rov_pose = self.rov_pose

                self.rov_odom = np.vstack((self.rov_pose, self.rov_twist))

                rov_odom = Odometry()
                rov_odom.header.frame_id = "map_ned"
                rov_odom.header.stamp = rospy.Time.now()
                rov_odom.pose.pose.position.x = self.rov_odom[0][0]
                rov_odom.pose.pose.position.y = self.rov_odom[1][0]
                rov_odom.pose.pose.position.z = self.rov_odom[2][0]
                quat = R.from_euler(
                    "xyz",
                    [self.rov_odom[3][0], self.rov_odom[4][0], self.rov_odom[5][0]],
                ).as_quat()
                rov_odom.pose.pose.orientation.x = quat[0]
                rov_odom.pose.pose.orientation.y = quat[1]
                rov_odom.pose.pose.orientation.z = quat[2]
                rov_odom.pose.pose.orientation.w = quat[3]
                rov_odom.twist.twist.linear.x = self.rov_odom[6][0]
                rov_odom.twist.twist.linear.y = self.rov_odom[7][0]
                rov_odom.twist.twist.linear.z = self.rov_odom[8][0]
                rov_odom.twist.twist.angular.x = self.rov_odom[9][0]
                rov_odom.twist.twist.angular.y = self.rov_odom[10][0]
                rov_odom.twist.twist.angular.z = self.rov_odom[11][0]

                self.rov_odom_pub.publish(rov_odom)
            # print(self.rov_pose)
        except Exception:
            rospy.logerr_throttle(
                10, "[BlueROV2][rov_pose_cb] Not receiving ROV's Position"
            )

    def rov_vel_cb(self, vel):
        """Callback function for the ROV's velocity

        Args:
            vel: TwistStamped message
        """
        try:
            self.rov_twist = np.zeros((6, 1))
            self.rov_twist[0][0] = vel.twist.linear.x
            self.rov_twist[1][0] = -vel.twist.linear.y
            self.rov_twist[2][0] = -vel.twist.linear.z
            # self.rov_twist[3][0] = vel.twist.angular.x
            # self.rov_twist[4][0] = -vel.twist.angular.y
            self.rov_twist[5][0] = -vel.twist.angular.z
            self.rov_twist[3][0] = 0.0
            self.rov_twist[4][0] = 0.0
            # self.rov_twist[5][0] = 0.0
            # print(self.rov_twist)
        except Exception:
            rospy.logerr_throttle(
                10, "[BlueROV2][rov_vel_cb] Not receiving ROV's velocity"
            )

    def store_sub_data(self, data, key):
        """Store subscriber data in a dictionary

        Args:
            data: Subscriber data
            key: Key to store the data
        """
        try:
            self.sub_data_dict[key] = data
        except Exception:
            rospy.logerr_throttle(
                10, "[BlueROV2][store_sub_data] Not receiving {} data".format(key)
            )

    def thrust_to_pwm(self, thrust):
        """Convert thrust values to pwm values

        Args:
            thrust: Thrust values

        Returns:
            pwm: PWM values
        """
        values = []
        thrust = thrust.flatten()

        try:
            for i in range(8):
                t = thrust[i] / 9.8
                t = np.round(t, 3)

                if t > 0.0:
                    p = np.interp(t, self.pos_thrusts, self.pos_pwm)
                elif t < 0.0:
                    p = np.interp(t, self.neg_thrusts, self.neg_pwm)
                else:
                    p = 1500

                values.append(round(p))

            pwm = values

        except Exception:
            rospy.logerr_throttle(
                10, "[BlueROV2][thrust_to_pwm] Error in thrust to pwm conversion."
            )
            pwm = [self.neutral_pwm for _ in range(8)]

        return pwm

    def calculate_pwm_from_thrust_curve(self, force):
        # The thrust curve is only accurate over the following force range,
        # so we restrict the input forces to that range
        min_force = -40.0
        max_force = 60.0

        force = np.clip(force, min_force, max_force)

        # Coefficients for the 4th-degree polynomial fit to the
        # thrust curve for the T200 run using a battery at 18v.
        # The polynomial was identified using Matlab's `fit` function.
        p00 = 1498
        p01 = 12.01
        p02 = -0.04731
        p03 = -0.002098
        p04 = 0.00002251

        pwm = p00 + p01 * force + p02 * force**2 + p03 * force**3 + p04 * force**4

        pwm = np.round(pwm)
        pwm = pwm.tolist()
        pwm = [int(i) for i in pwm]

        return pwm

    def arm(self):
        """Arm the vehicle"""
        rospy.wait_for_service("/mavros/cmd/arming")
        self.arm_srv(True)
        rospy.loginfo("[BlueROV2][arm] Arming vehicle")

        # Disarm is necessary when shutting down
        rospy.on_shutdown(self.disarm)

    def disarm(self):
        """Disarm the vehicle"""
        rospy.loginfo("[BlueROV2][disarm] Disarming vehicle")
        self.arm_srv(False)
        rospy.wait_for_service("/mavros/cmd/arming")
        self.arm_srv(False)

    def controller(self, joy):
        """Controller function to switch between manual and autonomous mode

        Args:
            joy: Joy message
        """
        buttons = joy.buttons

        # Switch into autonomous mode when button "A" is pressed
        # (Switches back into manual mode when the control sticks are moved)
        if buttons[0]:
            self.mode_flag = "auto"

        # set arm and disarm (disarm default)
        if buttons[6] == 1:  # "back" joystick button
            self.disarm()
        elif buttons[7] == 1:  # "start" joystick button
            self.arm()

        # set autonomous or manual control (manual control default)
        if self.mode_flag == "auto":
            if not self.rc_passthrough_flag:
                # Enable RC Passthrough Mode
                response = self.rc_passthrough_srv(True)
                if response.success:
                    self.rc_passthrough_flag = True
                    rospy.logwarn_throttle(
                        10, "[BlueROV2][controller] You are in AUTO mode!"
                    )
                    self.auto_control(joy)
                else:
                    msg = """
                    [BlueROV2][controller] Unable to switch to
                    RC Passthrough mode. Cannot enable AUTO mode!
                    """
                    rospy.logwarn_throttle(
                        10,
                        msg,
                    )
                    self.mode_flag = "manual"
            else:
                self.auto_control(joy)
        else:
            if self.rc_passthrough_flag:
                response = self.rc_passthrough_srv(False)
                self.rc_passthrough_flag = False
            # Disable RC Passthrough Mode
            rospy.logwarn_throttle(10, "[BlueROV2][controller] You are in MANUAL mode!")
            self.manual_control(joy)

    def auto_control(self, joy):
        """Function to control the ROV autonomously

        Args:
            joy: Joy message
        """
        # Switch out of autonomous mode if thumbstick input is detected
        # Grab the values of the control sticks
        axes = joy.axes
        control_sticks = axes[0:2] + axes[3:5]
        # Check if there is any input on the control sticks
        control_sticks = [abs(val) for val in control_sticks]
        if sum(control_sticks) > 0:
            # Set mode to manual
            self.mode_flag = "manual"
            return

        # if self.rov_pose is None or self.rov_twist is None:
        if self.rov_odom is None:
            rospy.logerr_throttle(
                10, "[BlueROV2][auto_contol] ROV odom not initialized"
            )
            return
        else:
            # self.rov_odom = np.vstack((self.rov_pose, self.rov_twist))
            x0 = self.rov_odom

        # x0 = np.array([[0., 0., 0., 0., 0., 0, 0., 0., 0., 0., 0., 0.]]).T
        xr = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T

        try:
            forces, wrench, converge_flag = self.mpc.run_mpc(x0, xr)
            if converge_flag:
                msg = """
                [BlueROV2][auto_control] ROV reached dock successfully!
                Disarming now...
                """
                rospy.loginfo_throttle(
                    10,
                    msg,
                )
                response = self.rc_passthrough_srv(False)
                self.rc_passthrough_flag = False
                if response.success:
                    self.disarm()
                self.mode_flag = "manual"
            else:
                wrench_frd = WrenchStamped()
                wrench_frd.header.frame_id = "base_link_frd"
                wrench_frd.header.stamp = rospy.Time()
                wrench_frd.wrench.force.x = wrench[0]
                wrench_frd.wrench.force.y = wrench[1]
                wrench_frd.wrench.force.z = wrench[2]
                wrench_frd.wrench.torque.x = wrench[3]
                wrench_frd.wrench.torque.y = wrench[4]
                wrench_frd.wrench.torque.z = wrench[5]
                self.mpc_wrench_pub.publish(wrench_frd)

                mpc_op = Float32MultiArray()
                dim = MultiArrayDimension()
                dim.label = "MPC Forces"
                dim.size = 8
                dim.stride = 1
                mpc_op.layout.dim.append(dim)
                mpc_op.data = [float(forces[i][0]) for i in range(8)]
                self.mpc_output.publish(mpc_op)

                # pwm = self.thrust_to_pwm(forces)
                pwm = self.calculate_pwm_from_thrust_curve(forces[:, 0])

                for i in range(len(pwm)):
                    if pwm[i] > self.deadzone_pwm[0] and pwm[i] < self.deadzone_pwm[1]:
                        pwm[i] = self.neutral_pwm

                for _ in range(len(pwm), 18):
                    pwm.append(OverrideRCIn.CHAN_NOCHANGE)
                    # pwm.append(self.neutral_pwm)

                for i in range(len(pwm)):
                    pwm[i] = max(
                        min(pwm[i], self.max_possible_pwm), self.min_possible_pwm
                    )

                self.mpc_pwm_pub.publish(pwm)
                self.control_pub.publish(pwm)
        except Exception as e:
            rospy.logerr_throttle(
                10, "[BlueROV2][auto_control] Error in MPC Computation" + str(e)
            )
            return

    def manual_control(self, joy):
        """Function to control the ROV manually using a joystick

        Args:
            joy: Joy message
        """
        axes = joy.axes
        buttons = joy.buttons

        self.mpc.mpc.reset()

        # Create a copy of axes as a list instead of a tuple so you can
        # modify the values.
        # RCOverrideOut message type also expects a list
        temp_axes = list(axes)
        temp_axes[3] *= -1  # fixes reversed yaw axis
        temp_axes[0] *= -1  # fixes reversed lateral L/R axis

        # Remap joystick commands [-1.0 to 1.0] to RC_override commands [1100 to 1900]
        adjusted_joy = [int(val * 300 + self.neutral_pwm) for val in temp_axes]
        override = [self.neutral_pwm for _ in range(18)]

        # print(adjusted_joy)

        # Remap joystick channels to correct ROV channel
        joy_mapping = [(0, 5), (1, 4), (3, 3), (4, 2), (7, 7)]

        for pair in joy_mapping:
            # print(pair)
            override[pair[1]] = adjusted_joy[pair[0]]

        if buttons[5] == 1:
            self.light_level += 100
        elif buttons[4] == 1:
            self.light_level -= 100

        override[8] = self.light_level

        # Cap the pwm value (limits the ROV velocity)
        for i in range(len(override)):
            override[i] = max(
                min(override[i], self.max_pwm_manual), self.min_pwm_manual
            )

        self.light_level = override[8]

        # Send joystick data as rc output into rc override topic
        self.control_pub.publish(override)

    def timer_cb(self, timer_event):
        """Timer callback function to publish the control data at a fixed rate

        Args:
            timer_event: Timer event
        """
        if self.override is not None:
            self.control_pub.publish(self.override)

    def run(self):
        """Main function to run the controller"""
        rate = rospy.Rate(80)

        while not rospy.is_shutdown():
            # Try to get joystick axes and button data
            try:
                joy = self.sub_data_dict["joy"]
                # Activate joystick control
                self.controller(joy)
            except Exception as error:
                rospy.logerr_throttle(
                    10, "[BlueROV2][run] Controller error:" + str(error)
                )

            rate.sleep()


if __name__ == "__main__":
    try:
        rospy.init_node("mission_control", anonymous=True)
    except KeyboardInterrupt:
        rospy.logwarn("Shutting down the node")

    obj = BlueROV2()
    obj.run()
