#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from collections import deque
import cv2
import random
import torch
from deep_classifier import DeepClassifier
import rospkg
import os

class DeepBehaviourModelWrapper:
    def __init__(self, path):
        self.camera_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        
        self.camera_sub = rospy.Subscriber(self.camera_topic,
                                            Image,
                                            self.camera_cb)
        self.image_pub = rospy.Publisher('test_img', Image, queue_size=10)
        self.frame_buffer = deque(maxlen=200)
        self.bridge = CvBridge()
        self.frame = np.ones((198, 198))
        self.model = DeepClassifier(input_state_size=1, path=model_path)
        self.mean = 127
        self.std = 77

    def camera_cb(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        cv_image = cv2.resize(cv_image, (198, 198))
        self.frame_buffer.append(cv_image)
        image_message = self.bridge.cv2_to_imgmsg(cv_image)
        self.image_pub.publish(image_message)
        
    def get_similarity(self, frame1, frame2):
        diff = frame1/255 - frame2/255
        diff = np.around(diff, 2)
        nonzero_elem = np.count_nonzero(diff)
        total = diff.shape[0]*diff.shape[1]
        
        return 1.-nonzero_elem/total
        
    def filter_frames(self, frames, size=8, threshold=0.8):
        similarities = []

        for idx in range(0, frames.shape[0]-1):
            similarity = self.get_similarity(frames[idx], frames[idx+1])
            similarities.append(similarity)

        similarities = np.array(similarities)
        dissimilar_ids = np.argwhere(similarities < threshold)
        dissimilarities = (similarities[dissimilar_ids]).flatten()

        if not list(dissimilar_ids):
            random_frame_id = random.randint(0, frames.shape[0]-1)
            random_frame = frames[random_frame_id]
            frames = np.array([random_frame]*size)
        
        else:
            if len(dissimilar_ids) < size:
                frames = frames[dissimilar_ids]
                last_frame = frames[-1]
                frames = list(frames)
                [frames.append(last_frame) for id in range(size-len(dissimilar_ids))]

            elif len(dissimilar_ids) == size:
                frames = frames[dissimilar_ids]

            else:
                indexes = np.argpartition(dissimilarities, size)[:size]
                indexes = np.sort(indexes)
                frames = frames[dissimilar_ids[indexes]]
        return frames
            
    def act(self):
        frames = self.filter_frames(np.array(self.frame_buffer))
        frames = (np.asarray(frames, dtype=np.float32)-self.mean)/self.std
        frames = torch.from_numpy(frames)
        frames = torch.movedim(frames, 1, 0)
        activity = torch.from_numpy(np.asarray([1, 0, 1, 0], dtype=np.float32))
        prediction = self.model.majority_vote((frames, activity))
        prediction = prediction
        print(prediction)
        


if __name__ == '__main__':
    
    rospy.init_node('deep_behaviour_model')
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('migrave_deep_behaviour_model')
    model_path = os.path.join(package_path, 'checkpoint')

    deep_behaviour_model = DeepBehaviourModelWrapper(path=model_path)
    
    rospy.sleep(1.0)
    
    try:
        while not rospy.is_shutdown():
            deep_behaviour_model.act()
            rospy.sleep(0.5)
    except rospy.ROSInterruptException as exc:
        print('Deep behaviour model wrapper exiting...')