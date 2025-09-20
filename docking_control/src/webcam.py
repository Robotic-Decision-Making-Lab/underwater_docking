#!/usr/bin/env python3

# This script creates a ROS node that captures video from a webcam
# and publishes it as a sensor_msgs/Image message.

import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


def webcam_publisher():
    """
    Captures video from the default webcam and publishes it to a ROS topic.
    """
    # Initialize the ROS node.
    # 'anonymous=True' ensures that your node has a unique name by adding a
    # random number to the end of it.
    rospy.init_node("webcam_publisher", anonymous=True)

    # Create a publisher.
    # This will publish messages of type sensor_msgs/Image on the topic
    # '/BlueROV2/video'.
    # The queue_size argument is a buffer that limits the number of messages
    # to store if the subscriber is not receiving them fast enough.
    image_pub = rospy.Publisher("/BlueROV2/video", Image, queue_size=10)

    # Create a CvBridge object to convert between OpenCV and ROS image formats.
    bridge = CvBridge()

    # Open the default webcam. The '0' argument typically refers to the default
    # built-in webcam. If you have multiple cameras, you might need to use 1, 2, etc.
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

    # Check if the webcam was opened successfully.
    if not cap.isOpened():
        rospy.logerr("Could not open webcam.")
        return

    # Set the publishing rate (e.g., 30 Hz).
    rate = rospy.Rate(30)

    rospy.loginfo("Webcam node started. Publishing video feed...")

    # Main loop that runs as long as ROS is active.
    while not rospy.is_shutdown():
        # Capture a single frame from the webcam.
        # 'ret' is a boolean that is True if the frame was read successfully.
        # 'frame' is the captured image.
        ret, frame = cap.read()

        if ret:
            try:
                # Convert the OpenCV image (BGR format) to a ROS Image message.
                # The encoding "bgr8" means 8-bit Blue-Green-Red channels.
                image_message = bridge.cv2_to_imgmsg(frame, "bgr8")

                # Publish the ROS Image message.
                image_pub.publish(image_message)

            except CvBridgeError as e:
                # Log any errors during the conversion process.
                rospy.logerr(e)

        # Sleep to maintain the desired publishing rate.
        rate.sleep()

    # When the node is shut down (e.g., by pressing Ctrl+C), release the webcam.
    cap.release()
    rospy.loginfo("Webcam node shut down and camera released.")


if __name__ == "__main__":
    try:
        webcam_publisher()
    except rospy.ROSInterruptException:
        # This exception is raised when Ctrl+C is pressed or the node is otherwise
        # shut down. Passing here allows the script to exit cleanly.
        pass
