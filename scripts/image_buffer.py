#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
import cv2
from cv_bridge import CvBridge
from collections import deque
import random
import torch
from torchvision.utils import draw_segmentation_masks


class ImageBuffer:
    def __init__(self, camera_topic, buffer_len=200, image_width=198, image_height=198):
        self.buffer_len = buffer_len
        self.camera_topic = camera_topic
        self.camera_sub = rospy.Subscriber(self.camera_topic,
                                           Image,
                                           self.__camera_cb)

        self.image_pub = rospy.Publisher('test_img', Image, queue_size=10)
        self.frame_buffer = deque(maxlen=buffer_len)
        self.bridge = CvBridge()
        self.frame = np.ones((image_width, image_height))
        self.is_camera_subscribed = True

    def switch_on(self):
        self.is_camera_subscribed = True

    def switch_off(self):
        self.is_camera_subscribed = False

    def reset(self):
        self.frame_buffer = deque(maxlen=self.buffer_len)

    def __camera_cb(self, msg):
        if self.is_camera_subscribed:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            cv_image = cv2.resize(cv_image, (198, 198))
            self.frame_buffer.append(cv_image)

    def __get_similarity(self, frame1, frame2):
        diff = frame1 / 255 - frame2 / 255
        diff = np.around(diff, 2)
        nonzero_elem = np.count_nonzero(diff)
        total = diff.shape[0] * diff.shape[1]
        return 1. - nonzero_elem / total

    def __augment(self, image):
        img = image.repeat(3, 1, 1).unsqueeze(dim=0)

        batch = self.transforms(img)
        output = self.seg_model(batch)['out']

        normalized_mask = torch.nn.functional.softmax(output, dim=1)
        boolean_mask = normalized_mask.argmax(1) == self.sem_class_to_idx['person']

        img = img.squeeze()

        background = np.expand_dims(np.ones(shape=(198, 198)) * 255, axis=0).astype('uint8')
        background = torch.from_numpy(background)
        background = background.repeat((3, 1, 1))

        foreground = draw_segmentation_masks(img, masks=~boolean_mask, alpha=1.0)
        background = draw_segmentation_masks(background, masks=boolean_mask, alpha=1.0)

        augmented_img = foreground + background

        return augmented_img[0].numpy()

    def __filter_frames(self, frames, size=8, threshold=0.8):
        similarities = []

        for idx in range(0, frames.shape[0] - 1):
            similarity = self.get_similarity(frames[idx], frames[idx + 1])
            similarities.append(similarity)

        similarities = np.array(similarities)
        dissimilar_ids = np.argwhere(similarities < threshold)
        dissimilarities = (similarities[dissimilar_ids]).flatten()

        if not list(dissimilar_ids):
            random_frame_id = random.randint(0, frames.shape[0] - 1)
            random_frame = frames[random_frame_id]
            frames = np.array([random_frame] * size)
            frames = frames[:, np.newaxis, :, :]

        else:
            if len(dissimilar_ids) < size:
                frames = frames[dissimilar_ids]
                last_frame = frames[-1]
                frames = list(frames)
                [frames.append(last_frame) for id in range(size - len(dissimilar_ids))]
                frames = np.array(frames)

            elif len(dissimilar_ids) == size:
                frames = frames[dissimilar_ids]

            else:
                indexes = np.argpartition(dissimilarities, size)[:size]
                indexes = np.sort(indexes)
                frames = frames[dissimilar_ids[indexes]]

        return frames

    def __augment_frames(self, frames):
        images = []
        for frame in frames:
            image = self.augment(torch.from_numpy(frame))
            images.append(image)

        return np.array(images)[:, np.newaxis, :, :]

    def get_frames(self, augment_frames=False):
        frames = self.__filter_frames(np.array(self.frame_buffer))

        if augment_frames:
            frames = self.augment_frames(frames)

        return frames
