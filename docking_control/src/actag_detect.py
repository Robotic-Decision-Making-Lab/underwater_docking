#!/usr/bin/env python3.10

import yaml

# from actag_code import AcTagDetection
from actag import AcTag
import rospy
import os
import cv2
import math
import numpy as np

# import video as video
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from tf2_ros import TransformException, TransformStamped
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros.transform_broadcaster import TransformBroadcaster
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R
from collections import deque


# Helper function that parses AcTag detection data
# and prints the results
def parse_and_print_results(detected_tags):
    num_detected_tags = len(detected_tags)
    print("Number of Detected Tags: ", num_detected_tags)

    for i in range(0, len(detected_tags)):
        print("==== Tag #" + str(i) + " ====")
        detected_tag = detected_tags[i]
        print("Tag ID: ", detected_tag[0])
        print("Corner Locations: \n", detected_tag[1])
        print("Corner Range & Azimuth Locations: \n", detected_tag[2])


# This helper function draws the detected tags onto the image
def visualize_decoded_tags(my_sonar_image, detected_tags):
    output_image = cv2.cvtColor(my_sonar_image, cv2.COLOR_GRAY2RGB)
    for detected_tag in detected_tags:
        # Extract corner points
        corner_locs = detected_tag[1]
        ptA = (corner_locs[0][0], corner_locs[0][1])
        ptB = (corner_locs[1][0], corner_locs[1][1])
        ptC = (corner_locs[2][0], corner_locs[2][1])
        pdf = (corner_locs[3][0], corner_locs[3][1])

        # Reverse x and y to get the correct orientation with cv2.imshow()
        ptA = (ptA[1], ptA[0])
        ptB = (ptB[1], ptB[0])
        ptC = (ptC[1], ptC[0])
        pdf = (pdf[1], pdf[0])

        # Draw the bounding box of the AcTag Square
        color = (0, 255, 0)
        cv2.line(output_image, ptA, ptB, color, 1)
        cv2.putText(output_image, "1", ptA, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.line(output_image, ptB, ptC, color, 1)
        cv2.putText(output_image, "2", ptB, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.line(output_image, ptC, pdf, color, 1)
        cv2.putText(output_image, "3", ptC, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.line(output_image, pdf, ptA, color, 1)
        cv2.putText(output_image, "4", pdf, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Put the Tag ID in the center
        center = (int((ptA[0] + ptC[0]) / 2), int((ptA[1] + ptC[1]) / 2))
        cv2.putText(
            output_image,
            "#" + str(detected_tag[0]),
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    return output_image


class TagDetector:
    def __init__(self):
        """This class is used to detect AcTags and estimate the pose
        of the ROV.
        """

        # print("Initializing AcTag Detector")

        self.service_flag = False
        self.bridge = CvBridge()

        self.filter_len = 20
        self.pose_buffer = deque(maxlen=self.filter_len)

        # self.detector = AcTagDetection()

        # print("Initializing AcTag Detector1")

        # Initialize the AcTag Detector
        self.detector = AcTag(
            min_range=0.1,
            max_range=1.5,
            horizontal_aperture=1.0472,
            tag_family="AcTag24h10",
            tag_size=0.130628571428644,
            quads_use_same_random_vals=True,
        )

        # Offset from the ROV's center of gravity (COG) to the camera center
        self.camera_offset = [0.16, -0.06]

        # Provide access to TF2
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        self.tf_broadcaster = TransformBroadcaster()

        self.initialize_subscribers_publishers()

        # self.load_camera_config()

        # rospy.Timer(rospy.Duration(0.1), self.marker_detection)

    def load_camera_config(self):
        """This function is used to load the camera calibration parameters."""
        cwd = os.path.dirname(__file__)
        filename = cwd + "/../config/in_air/ost.yaml"
        f = open(filename, "r")
        camera_params = yaml.load(f.read(), Loader=yaml.SafeLoader)
        self.cam_mat = np.array(
            camera_params["camera_matrix"]["data"], np.float32
        ).reshape(3, 3)
        self.proj_mat = np.array(
            camera_params["projection_matrix"]["data"], np.float32
        ).reshape(3, 4)
        self.dist_mat = np.array(
            camera_params["distortion_coefficients"]["data"], np.float32
        ).reshape(1, 5)

    def initialize_subscribers_publishers(self):
        """This function is used to initialize the subscribers and publishers."""
        # self.image_sub = rospy.Subscriber(
        #     "/BlueROV2/video", Image, self.callback_image, queue_size=1
        # )
        self.sonar_sub = rospy.Subscriber(
            "/oculus/drawn_sonar_rect", Image, self.callback_sonar, queue_size=1
        )
        # self.image_pub = rospy.Publisher(
        #     "/docking_control/marker_detection", Image, queue_size=1
        # )
        # self.vision_pose_pub = rospy.Publisher(
        #     "/docking_control/vision_pose/pose", PoseStamped, queue_size=1
        # )

    def rotation_matrix_to_euler(self, R):
        """This function is used to convert a rotation matrix to euler angles.

        Args:
            R: Rotation matrix
        """

        def isRotationMatrix(R):
            Rt = np.transpose(R)
            shouldBeIdentity = np.dot(Rt, R)
            mat_eye = np.identity(3, dtype=R.dtype)
            n = np.linalg.norm(mat_eye - shouldBeIdentity)
            return n < 1e-6

        assert isRotationMatrix(R)

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z], np.float32)

    def callback_image(self, data):
        """This function is used to receive the image from the camera.

        Args:
            data: Image data
        """
        try:
            self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_sonar(self, data):
        """This function is used to receive the sonar image.

        Args:
            data: Image data
        """
        try:
            # print("Received sonar image")
            sonar_frame = self.bridge.imgmsg_to_cv2(data, desired_encoding="rgb8")
            sonar_image = cv2.cvtColor(sonar_frame, cv2.COLOR_BGR2GRAY)
            # print(sonar_image.shape)

            # Detect tags in the image
            detected_tags = self.detector.run_detection(sonar_image)

            # Parse and print results
            parse_and_print_results(detected_tags)

            # Visualize decoded tags on the original image
            output_image = visualize_decoded_tags(sonar_image, detected_tags)
            cv2.imshow("Detected Tags", output_image)

            cv2.waitKey(1)

        except CvBridgeError as e:
            print(e)

    def marker_detection(self, timerEvent):
        """This function is used to detect the ArUco markers and
        estimate the pose of the ROV.

        Args:
            timerEvent: ROS timer event

        Returns:
            None
        """
        try:
            frame = self.image
        except Exception:
            print("[Aruco][marker_detection] Not receiving any image")
            return

        try:
            dist_mtx = self.dist_mat
            camera_mtx = self.cam_mat
        except Exception:
            print("[Aruco][marker_detection] Not receiving camera info")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        aruco_dict = cv2.aruco.getPredefinedDictionary(self.selected_dict)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

        try:
            # Marker detection
            corners, ids, rejectedImgPoints = detector.detectMarkers(gray)
        except BaseException:
            print("[Aruco][marker_detection] Error detecting markers")
            return

        np.array(corners)

        # If at least one marker is detected, then continue
        if ids is not None:
            # rospy.logwarn("Markers are present")

            detected_marker_ids = []
            des_marker_corners = []

            # Loop through the detected markers
            for i, j in enumerate(ids):
                # If the detected marker is not one of the desired markers, then ignore
                if j[0] in self.desired_markers:
                    # rospy.logwarn("Desired markers are present")
                    detected_marker_ids.append(j[0])
                    des_marker_corners.append(corners[i])

            detected_marker_ids = np.array(detected_marker_ids)
            des_marker_corners = np.array(des_marker_corners)

            cv2.aruco.drawDetectedMarkers(
                frame, des_marker_corners, detected_marker_ids
            )

            # Marker Pose Estimation
            i = 0
            if detected_marker_ids.shape[0] > 0:
                # for i in range(des_marker_corners.shape[0]):
                marker_id = detected_marker_ids[i]
                marker_id_str = "marker_{0}".format(marker_id)
                marker_size = self.marker_size[marker_id]

                marker_points = np.array(
                    [
                        [-marker_size / 2, marker_size / 2, 0],
                        [marker_size / 2, marker_size / 2, 0],
                        [marker_size / 2, -marker_size / 2, 0],
                        [-marker_size / 2, -marker_size / 2, 0],
                    ],
                    dtype=np.float32,
                )

                _, rvec, tvec = cv2.solvePnP(
                    marker_points, des_marker_corners[i], camera_mtx, dist_mtx
                )

                rvec = np.array(rvec).T
                tvec = np.array(tvec).T

                # Calculates rodrigues matrix
                rmat, _ = cv2.Rodrigues(rvec)
                # Convert rmat to euler
                # rvec = self.rotation_matrix_to_euler(rmat)
                rvec = R.from_matrix(rmat).as_euler("xyz")

                cv2.drawFrameAxes(frame, camera_mtx, dist_mtx, rvec, tvec, 0.1)

                quat = R.from_matrix(rmat).as_quat()

                tf_cam_to_marker = TransformStamped()
                tf_cam_to_marker.header.frame_id = "camera_link"
                tf_cam_to_marker.child_frame_id = marker_id_str
                tf_cam_to_marker.header.stamp = rospy.Time.now()
                tf_cam_to_marker.transform.translation.x = tvec[0][0]
                tf_cam_to_marker.transform.translation.y = tvec[0][1]
                tf_cam_to_marker.transform.translation.z = tvec[0][2]
                tf_cam_to_marker.transform.rotation.x = quat[0]
                tf_cam_to_marker.transform.rotation.y = quat[1]
                tf_cam_to_marker.transform.rotation.z = quat[2]
                tf_cam_to_marker.transform.rotation.w = quat[3]

                # transform lookup: base_link -> camera_link
                try:
                    tf_base_to_camera = self.tf_buffer.lookup_transform(
                        "base_link", "camera_link", rospy.Time()
                    )
                except TransformException as e:
                    rospy.logwarn(
                        "[Aruco][marker_detection] Transform unavailable: {0}".format(e)
                    )
                    return

                # transform lookup: marker -> map
                try:
                    tf_marker_to_map = self.tf_buffer.lookup_transform(
                        marker_id_str, "map", rospy.Time()
                    )
                except TransformException as e:
                    rospy.logwarn(
                        "[Aruco][marker_detection] Transform unavailable: {0}".format(e)
                    )
                    return

                def transform_to_arr(t):
                    arr = np.eye(4)
                    arr[:3, 3] = (
                        t.transform.translation.x,
                        t.transform.translation.y,
                        t.transform.translation.z,
                    )
                    arr[:3, :3] = R.from_quat(
                        [
                            t.transform.rotation.x,
                            t.transform.rotation.y,
                            t.transform.rotation.z,
                            t.transform.rotation.w,
                        ]
                    ).as_matrix()
                    return arr

                def arr_to_pose(arr, frame_id):
                    pose = PoseStamped()
                    pose.header.frame_id = frame_id
                    pose.header.stamp = rospy.Time.now()
                    pose.pose.position.x = arr[0, 3]
                    pose.pose.position.y = arr[1, 3]
                    pose.pose.position.z = arr[2, 3]
                    quat = R.from_matrix(arr[:3, :3]).as_quat()
                    pose.pose.orientation.x = quat[0]
                    pose.pose.orientation.y = quat[1]
                    pose.pose.orientation.z = quat[2]
                    pose.pose.orientation.w = quat[3]
                    return pose

                tf_base_to_marker = transform_to_arr(
                    tf_base_to_camera
                ) @ transform_to_arr(tf_cam_to_marker)
                tf_base_to_map = tf_base_to_marker @ transform_to_arr(tf_marker_to_map)

                rov_pose = arr_to_pose(np.linalg.inv(tf_base_to_map), "map")

                rov_orientation = R.from_quat(
                    [
                        rov_pose.pose.orientation.x,
                        rov_pose.pose.orientation.y,
                        rov_pose.pose.orientation.z,
                        rov_pose.pose.orientation.w,
                    ]
                ).as_euler("xyz")

                rov_pose_arr = np.array(
                    [
                        rov_pose.pose.position.x,
                        rov_pose.pose.position.y,
                        rov_pose.pose.position.z,
                        rov_orientation,
                    ]
                )

                self.pose_buffer.append(rov_pose_arr)

                if len(self.pose_buffer) < self.pose_buffer.maxlen:
                    return

                def moving_average(data):
                    data = np.array(data)
                    weights = np.arange(len(data)) + 1

                    # Apply the LWMA filter and return
                    return np.array(
                        [
                            np.sum(np.prod(np.vstack((axis, weights)), axis=0))
                            / np.sum(weights)
                            for axis in data.T
                        ]
                    )

                filtered_pose = moving_average(self.pose_buffer)

                rov_pose.pose.position.x = filtered_pose[0]
                rov_pose.pose.position.y = filtered_pose[1]
                rov_pose.pose.position.z = filtered_pose[2]

                quat = R.from_euler("xyz", filtered_pose[3:][0]).as_quat()
                rov_pose.pose.orientation.x = quat[0]
                rov_pose.pose.orientation.y = quat[1]
                rov_pose.pose.orientation.z = quat[2]
                rov_pose.pose.orientation.w = quat[3]

                self.vision_pose_pub.publish(rov_pose)

        cv2.circle(
            frame,
            (int(frame.shape[1] / 2), int(frame.shape[0] / 2)),
            radius=5,
            color=(255, 0, 0),
        )
        cv2.imshow("marker", frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    rospy.init_node("tag_detector", anonymous=True)
    obj = TagDetector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting down the node")
