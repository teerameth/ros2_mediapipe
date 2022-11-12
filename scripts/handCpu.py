#!/usr/bin/env -S HOME=${HOME} ${HOME}/.virtualenvs/cv/bin/python

from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray

import cv2
import numpy as np
import mediapipe as mp

cv_bridge = CvBridge()
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

class MediapipeHand(Node):
    def __init__(self):
        super().__init__('mediapipe_hand_cpu')
        self.img_sub = self.create_subscription(Image, '/camera/image', self.image_sub_callback, 10)
        self.hand_pub = self.create_publisher(Float32MultiArray, '/camera/hand', 10)
    def image_sub_callback(self, imgmsg):
        img = cv_bridge.imgmsg_to_cv2(imgmsg, desired_encoding='passthrough')
        results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        all_landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i, point in enumerate(hand_landmarks.landmark):
                    all_landmarks.append([point.x, point.y, point.z])
        landmark_msg = Float32MultiArray()
        landmark_msg.data = all_landmarks
        self.hand_pub.publish(landmark_msg)