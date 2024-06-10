#!/usr/bin/env python3

import cv2
import gi
import numpy as np
import rospy
from cv_bridge import CvBridge

from sensor_msgs.msg import Image

gi.require_version("Gst", "1.0")
from gi.repository import Gst


class Video:
    """BlueRov video capture class constructor

    Attributes:
        port (int): Video UDP port
        video_codec (string): Source h264 parser
        video_decode (string): Transform YUV (12bits) to BGR (24bits)
        video_pipe (object): GStreamer top-level pipeline
        video_sink (object): Gstreamer sink element
        video_sink_conf (string): Sink configuration
        video_source (string): Udp source ip and port
    """

    def __init__(self):
        """Summary

        Args:
            port (int, optional): UDP port
        """
        self.cvbridge = CvBridge()

        Gst.init(None)
        registry = Gst.Registry.get()

        nvcodec_plugin = Gst.Registry.lookup_feature(registry, "nvcodec")
        if nvcodec_plugin is not None:
            Gst.PluginFeature.set_rank(nvcodec_plugin, 0)

        libav_plugin = Gst.Registry.lookup_feature(registry, "libav")
        if libav_plugin is not None:
            Gst.PluginFeature.set_rank(libav_plugin, Gst.Rank.PRIMARY)

        self.port = rospy.get_param("/video_feed/video_udp_port")
        self._frame = None

        # UDP video stream (:5600)
        self.video_source = "udpsrc port={}".format(self.port)

        # RTSP video stream
        # self.video_source = 'rtspsrc location=rtsp://192.168.2.2:8554/video_stream__dev_video2 latency=0'

        # [Rasp raw image](http://picamera.readthedocs.io/en/release-0.7/recipes2.html#raw-image-capture-yuv-format)
        # Cam -> CSI-2 -> H264 Raw (YUV 4-4-4 (12bits) I420)
        self.video_codec = (
            "! application/x-rtp, payload=96 ! rtph264depay ! h264parse ! avdec_h264"
        )
        # self.video_codec = '! application/x-rtp, payload=96 ! rtph264depay ! h264parse ! v4l2h264dec'
        # self.video_codec = '! application/x-rtp, payload=96 ! rtpjpegdepay ! jpegdec'

        # Python don't have nibble, convert YUV nibbles (4-4-4) to OpenCV standard BGR bytes (8-8-8)
        self.video_decode = "! videoconvert ! video/x-raw,format=(string)BGR"

        # Create a sink to get data
        self.video_sink_conf = (
            "! appsink emit-signals=true sync=false max-buffers=2 drop=true"
        )

        self.video_pipe = None
        self.video_sink = None
        self.video_publisher = rospy.Publisher("/BlueROV2/video", Image, queue_size=1)
        self.run()

    def start_gst(self, config=None):
        """ Start gstreamer pipeline and sink
        Pipeline description list e.g:
            [
                'videotestsrc ! decodebin', \
                '! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert',
                '! appsink'
            ]

        Args:
            config (list, optional): Gstreamer pileline description list
        """

        if not config:
            config = [
                "videotestsrc ! decodebin",
                "! videoconvert ! video/x-raw,format=(string)BGR ! videoconvert",
                "! appsink",
            ]

        command = " ".join(config)
        self.video_pipe = Gst.parse_launch(command)
        self.video_pipe.set_state(Gst.State.PLAYING)
        self.video_sink = self.video_pipe.get_by_name("appsink0")

    @staticmethod
    def gst_to_opencv(sample):
        """Transform byte array into np array

        Args:
            sample (TYPE): Description

        Returns:
            TYPE: Description
        """
        buf = sample.get_buffer()
        caps = sample.get_caps()
        array = np.ndarray(
            (
                caps.get_structure(0).get_value("height"),
                caps.get_structure(0).get_value("width"),
                3,
            ),
            buffer=buf.extract_dup(0, buf.get_size()),
            dtype=np.uint8,
        )
        return array

    def frame(self):
        """Get Frame

        Returns:
            iterable: bool and image frame, cap.read() output
        """
        return self._frame

    def frame_available(self):
        """Check if frame is available

        Returns:
            bool: true if frame is available
        """
        return not isinstance(self._frame, type(None))

    def run(self):
        """Get frame to update _frame"""

        self.start_gst(
            [
                self.video_source,
                self.video_codec,
                self.video_decode,
                self.video_sink_conf,
            ]
        )

        self.video_sink.connect("new-sample", self.callback)

    def callback(self, sink):
        sample = sink.emit("pull-sample")
        new_frame = self.gst_to_opencv(sample)
        ros_image = self.cvbridge.cv2_to_imgmsg(new_frame, "bgr8")
        self.video_publisher.publish(ros_image)
        self._frame = new_frame

        return Gst.FlowReturn.OK


if __name__ == "__main__":
    rospy.init_node("video_feed", anonymous=True)
    video = Video()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting down the node")
