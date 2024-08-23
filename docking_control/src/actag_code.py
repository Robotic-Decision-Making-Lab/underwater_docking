#!/usr/bin/env python3.12

from actag import AcTag
# import rospy


class AcTagDetection:
  def __init__(self):
      # Initialize the AcTag Detector
      self.detector = AcTag(
          min_range=0.1,
          max_range=1.5,
          horizontal_aperture=1.0472,
          tag_family="AcTag24h10",
          tag_size=0.130628571428644,
          quads_use_same_random_vals=True,
      )

  def detect_tags(self, sonar_image):
      # Detect AcTags in the sonar image
      detected_tags = self.detector.detect_tags(sonar_image)
      return detected_tags


# if __name__ == "__main__":
#     rospy.init_node("actag_detection", anonymous=True)
#     obj = AcTagDetection()
#     try:
#         rospy.spin()
#     except KeyboardInterrupt:
#         print("shutting down the node")
