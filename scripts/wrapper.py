#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
from collections import deque
import random
import torch
from deep_classifier import DeepClassifier
import rospkg
from torchvision.utils import draw_segmentation_masks
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights

class DeepBehaviourModelWrapper:
    def __init__(self, path):
        self.camera_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        
        self.camera_sub = rospy.Subscriber(self.camera_topic,
                                            Image,
                                            self.camera_cb)
        self.image_pub = rospy.Publisher('test_img', Image, queue_size=10)
        self.frame_buffer = deque(maxlen=150)
        self.bridge = CvBridge()
        self.frame = np.ones((198, 198))
        self.model = DeepClassifier(input_state_size=1, path=model_path)
        self.mean = 127
        self.std = 77
        self.class_map = {0:'dif2', 1:'diff3', 2:'feedback'}
        
        weights = FCN_ResNet50_Weights.DEFAULT
        self.transforms = weights.transforms(resize_size=None)
        self.sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
        self.seg_model = fcn_resnet50(weights=weights, progress=False)
        self.seg_model = self.seg_model.eval()

    def camera_cb(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        cv_image = cv2.resize(cv_image, (198, 198))
        self.frame_buffer.append(cv_image)
        
    def get_similarity(self, frame1, frame2):
        diff = frame1/255 - frame2/255
        diff = np.around(diff, 2)
        nonzero_elem = np.count_nonzero(diff)
        total = diff.shape[0]*diff.shape[1]
        return 1.-nonzero_elem/total
        
    def augment(self, image):
        img = image.repeat(3, 1, 1).unsqueeze(dim=0)

        batch = self.transforms(img)
        output = self.seg_model(batch)['out']

        normalized_mask = torch.nn.functional.softmax(output, dim=1)
        boolean_mask = normalized_mask.argmax(1) == self.sem_class_to_idx['person']

        img = img.squeeze()

        background = np.expand_dims(np.ones(shape=(198, 198))*255, axis=0).astype('uint8')
        background = torch.from_numpy(background)
        background = background.repeat((3, 1, 1))

        foreground = draw_segmentation_masks(img, masks=~boolean_mask, alpha=1.0)
        background = draw_segmentation_masks(background, masks=boolean_mask, alpha=1.0)

        augmented_img = foreground + background

        return augmented_img[0].numpy()
        
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
            frames = frames[:, np.newaxis, :, :]
        
        else:
            if len(dissimilar_ids) < size:
                frames = frames[dissimilar_ids]
                last_frame = frames[-1]
                frames = list(frames)
                [frames.append(last_frame) for id in range(size-len(dissimilar_ids))]
                frames = np.array(frames)

            elif len(dissimilar_ids) == size:
                frames = frames[dissimilar_ids]

            else:
                indexes = np.argpartition(dissimilarities, size)[:size]
                indexes = np.sort(indexes)
                frames = frames[dissimilar_ids[indexes]]
                
        return frames
    
    def augment_frames(self, frames):
        images = []
        for frame in frames:
            image = self.augment(torch.from_numpy(frame))
            images.append(image)
        
        return np.array(images)
            
    def act(self):
        frames = self.filter_frames(np.array(self.frame_buffer))
        
        frames = self.augment_frames(frames)
        image = np.concatenate([frame for frame in frames], axis=1)
        frames = frames[np.newaxis,:, :, :]
        
        frames = (np.asarray(frames, dtype=np.float32)-self.mean)/self.std        
        frames = torch.from_numpy(frames)
        #frames = torch.movedim(frames, 1, 0)
        activity = torch.from_numpy(np.asarray([1, 1, 0, 0], dtype=np.float32))
        prediction = self.model.predict((frames, activity))
        
        print(prediction)
        
        #for i, pred in enumerate(predictions):
        #    image = cv2.putText(image, self.class_map[int(pred)], (i*198, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        
        image = cv2.putText(image, self.class_map[int(prediction)], (i*198, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        image_message = self.bridge.cv2_to_imgmsg(image)
        self.image_pub.publish(image_message)
        


if __name__ == '__main__':
    
    rospy.init_node('deep_behaviour_model')
    rospack = rospkg.RosPack()
    package_path = rospack.get_path('migrave_deep_behaviour_model')

    deep_behaviour_model = DeepBehaviourModelWrapper(path=package_path)
    
    rospy.sleep(1.0)
    
    try:
        while not rospy.is_shutdown():
            deep_behaviour_model.act()
            rospy.sleep(5.0)
    except rospy.ROSInterruptException as exc:
        print('Deep behaviour model wrapper exiting...')
