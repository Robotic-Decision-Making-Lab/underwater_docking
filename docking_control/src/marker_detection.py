#!/usr/bin/env python3

import yaml
import rospy
import os
import cv2
import math
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from collections import deque


# Marker dictionary
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    # Add other dictionaries if needed
}


class Aruco:
    def __init__(self):
        """
        This class detects ArUco markers, calculates the pose of the camera
        relative to the marker, and then computes the ROV's pose in the map frame.
        """
        # --- Parameters ---
        self.desired_markers = [2, 3, 4, 7, 8, 9, 10, 11, 14]
        self.marker_size = {
            1: 0.20,
            2: 0.10,
            3: 0.05,
            4: 0.15,
            7: 0.05,
            8: 0.10,
            9: 0.10,
            10: 0.10,
            11: 0.10,
            14: 0.1524,  # 6 inches in meters
        }
        self.filter_len = 20

        # --- Vision Processing ---
        self.bridge = CvBridge()
        self.pose_buffer = deque(maxlen=self.filter_len)
        self.load_camera_config()

        # Initialize the ArUco detector once
        aruco_dict_type = ARUCO_DICT.get("DICT_6X6_50", cv2.aruco.DICT_6X6_50)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

        # --- ROS Setup ---
        self.image = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        self.initialize_subscribers_publishers()
        rospy.Timer(rospy.Duration(0.1), self.marker_detection_callback)

    def load_camera_config(self):
        """Loads camera calibration parameters from a YAML file."""
        try:
            # Use os.path.realpath to get the correct path even with symlinks
            cwd = os.path.dirname(os.path.realpath(__file__))
            filename = os.path.join(cwd, "../config/in_air/ost.yaml")
            with open(filename, "r") as f:
                camera_params = yaml.safe_load(f)
            self.cam_mat = np.array(
                camera_params["camera_matrix"]["data"], dtype=np.float32
            ).reshape(3, 3)
            self.dist_mat = np.array(
                camera_params["distortion_coefficients"]["data"], dtype=np.float32
            ).reshape(1, 5)
            rospy.loginfo("Camera configuration loaded successfully.")
        except (IOError, yaml.YAMLError, KeyError) as e:
            rospy.logfatal(f"Failed to load camera configuration: {e}")
            rospy.signal_shutdown("Error loading camera config")

    def initialize_subscribers_publishers(self):
        """Initializes ROS subscribers and publishers."""
        # Increased buff_size for high-res images to prevent lag
        self.image_sub = rospy.Subscriber(
            "/BlueROV2/video", Image, self.image_callback, queue_size=1, buff_size=2**24
        )
        self.image_pub = rospy.Publisher(
            "/docking_control/marker_detection", Image, queue_size=1
        )
        self.vision_pose_pub = rospy.Publisher(
            "/docking_control/vision_pose/pose", PoseStamped, queue_size=1
        )

    def image_callback(self, data):
        """Receives and stores the latest image from the camera."""
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"[Aruco] CvBridge Error: {e}")

    def marker_detection_callback(self, timer_event):
        """
        Main loop for detecting ArUco markers and estimating the ROV's pose.
        This function is called by a ROS Timer.
        """
        if self.image is None:
            rospy.logwarn_throttle(5, "[Aruco] No image received yet.")
            return

        frame = self.image.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Marker detection
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is not None:
            # Filter for desired markers and find the one with the largest area
            valid_markers = []
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in self.desired_markers:
                    # Calculate the area of the detected marker contour
                    area = cv2.contourArea(corners[i][0])
                    valid_markers.append({"id": marker_id, "corners": corners[i], "area": area})

            # Draw all detected markers on the frame
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            if valid_markers:
                # Select the marker with the largest area on screen (closest/most central)
                best_marker = max(valid_markers, key=lambda m: m["area"])

                # Estimate pose using only the single best marker
                try:
                    self.estimate_and_publish_pose(frame, best_marker["id"], best_marker["corners"])
                except Exception as e:
                    rospy.logerr_throttle(2, f"[Aruco] Error in pose estimation: {e}")

        # --- ALWAYS SHOW THE IMAGE WINDOW ---
        # Draw circle at the center of the image
        height, width = frame.shape[:2]
        center = (width // 2, height // 2)
        cv2.circle(frame, center, 5, (0, 255, 0), -1)
        cv2.imshow("Marker Detection", frame)
        cv2.waitKey(1) # This is crucial for the window to update

        # Also publish the annotated image to a ROS topic
        self.publish_annotated_image(frame)


    def estimate_and_publish_pose(self, frame, marker_id, marker_corners):
        """Estimates pose from a single marker and publishes it."""
        marker_size = self.marker_size.get(marker_id)
        if marker_size is None:
            rospy.logwarn(f"No size defined for marker ID {marker_id}. Skipping.")
            return

        # Define the 3D points of the marker in its own coordinate system
        marker_points = np.array([
                [-marker_size / 2,  marker_size / 2, 0],
                [ marker_size / 2,  marker_size / 2, 0],
                [ marker_size / 2, -marker_size / 2, 0],
                [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)

        # Get pose of marker in camera frame
        _, rvec, tvec = cv2.solvePnP(
            marker_points, marker_corners, self.cam_mat, self.dist_mat
        )

        # Draw the marker's axes on the image for debugging
        # IMPORTANT: Use the original rvec from solvePnP
        cv2.drawFrameAxes(frame, self.cam_mat, self.dist_mat, rvec, tvec, 0.1)

        # --- Transform Calculation ---
        # T_cam_marker: Transform from camera frame to marker frame
        rmat, _ = cv2.Rodrigues(rvec)

        t_cam_marker = np.eye(4)
        t_cam_marker[:3, :3] = rmat
        t_cam_marker[:3, 3] = tvec.flatten()

        # Get required transforms from the TF tree
        try:
            # Use rospy.Time(0) to get the latest available transform
            tf_base_to_cam_msg = self.tf_buffer.lookup_transform("base_link", "camera_link", rospy.Time(0), rospy.Duration(0.1))
            tf_marker_to_map_msg = self.tf_buffer.lookup_transform(f"marker_{marker_id}", "map", rospy.Time(0), rospy.Duration(0.1))
        except TransformException as e:
            rospy.logwarn(f"[Aruco] Transform lookup failed: {e}")
            return

        # Convert TF messages to 4x4 matrices
        t_base_cam = self.transform_to_matrix(tf_base_to_cam_msg)
        t_marker_map = self.transform_to_matrix(tf_marker_to_map_msg)

        # Calculate the ROV's pose in the map frame
        # Chain: T_map_base = inv(T_base_map)
        # T_base_map = T_base_cam * T_cam_marker * T_marker_map
        t_base_map = t_base_cam @ t_cam_marker @ t_marker_map
        # t_map_base = np.linalg.inv(t_base_map)
        t_map_base = t_base_map

        # --- Filtering and Publishing ---
        rov_position = t_map_base[:3, 3]
        rov_orientation_euler = R.from_matrix(t_map_base[:3, :3]).as_euler('xyz')

        # IMPORTANT: Concatenate into a flat 1D array for the filter
        rov_pose_arr = np.concatenate((rov_position, rov_orientation_euler))

        self.pose_buffer.append(rov_pose_arr)

        if len(self.pose_buffer) < self.filter_len:
            # Wait for the buffer to fill before publishing
            return

        # Apply the weighted moving average filter
        filtered_pose_arr = self.weighted_moving_average(self.pose_buffer)

        # Create and publish the final PoseStamped message
        rov_pose_msg = PoseStamped()
        rov_pose_msg.header.stamp = rospy.Time.now()
        rov_pose_msg.header.frame_id = "map"
        rov_pose_msg.pose.position.x = filtered_pose_arr[0]
        rov_pose_msg.pose.position.y = filtered_pose_arr[1]
        rov_pose_msg.pose.position.z = filtered_pose_arr[2]

        filtered_quat = R.from_euler('xyz', filtered_pose_arr[3:]).as_quat()
        rov_pose_msg.pose.orientation.x = filtered_quat[0]
        rov_pose_msg.pose.orientation.y = filtered_quat[1]
        rov_pose_msg.pose.orientation.z = filtered_quat[2]
        rov_pose_msg.pose.orientation.w = filtered_quat[3]

        self.vision_pose_pub.publish(rov_pose_msg)

    def weighted_moving_average(self, data):
        """
        Applies a Linearly Weighted Moving Average (LWMA) filter.
        More recent data points are given higher weight.
        """
        data = np.array(data)
        weights = np.arange(1, len(data) + 1)

        # Calculate weighted average for each column (x, y, z, r, p, y)
        return np.sum(data * weights[:, np.newaxis], axis=0) / np.sum(weights)

    def transform_to_matrix(self, t):
        """Converts a geometry_msgs/TransformStamped to a 4x4 NumPy matrix."""
        translation = t.transform.translation
        rotation = t.transform.rotation

        mat = np.eye(4)
        mat[:3, 3] = (translation.x, translation.y, translation.z)
        mat[:3, :3] = R.from_quat(
            [rotation.x, rotation.y, rotation.z, rotation.w]
        ).as_matrix()
        return mat

    def publish_annotated_image(self, frame):
        """Publishes the annotated image for debugging."""
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "bgr8"))
        except CvBridgeError as e:
            rospy.logerr(f"Failed to publish annotated image: {e}")

if __name__ == "__main__":
    rospy.init_node("marker_detection_node")
    try:
        aruco_detector = Aruco()
        rospy.spin()
    except (rospy.ROSInterruptException, KeyboardInterrupt):
        print("Shutting down the marker detection node.")
    finally:
        # Clean up the OpenCV window when the node is shut down
        cv2.destroyAllWindows()
