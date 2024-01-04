#!/usr/bin/env python3
import os
from csv import DictWriter
from datetime import datetime

import rospy
import numpy as np
from sensor_msgs.msg import Image
import cv2
from migrave_deep_behaviour_model.msg import GameAction
from migrave_ros_msgs.msg import GameActivity, GamePerformance
import random
import torch
from std_msgs.msg import Int32, Bool
from deep_classifier import DeepClassifier
import rospkg
from image_buffer import ImageBuffer
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights


class DeepBehaviourModelWrapper:
    def __init__(self, package_path, log_path, model_name):
        camera_topic = rospy.get_param('~image_topic', '/camera/color/image_raw')
        self.image_pub = rospy.Publisher('test_img', Image, queue_size=10)
        self.model = DeepClassifier(input_state_size=8,
                                    input_modalities=2,
                                    path=package_path,
                                    model_name=model_name)
        self.image_width = 198
        self.image_height = 198
        self.buffer = ImageBuffer(camera_topic=camera_topic,
                                  image_width=self.image_width,
                                  image_height=self.image_height)
        self.mean = 127
        self.std = 77
        self.frame = np.ones((self.image_width, self.image_height))
        self.output_path = log_path
        
        weights = FCN_ResNet50_Weights.DEFAULT
        self.transforms = weights.transforms(resize_size=None)
        self.sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(weights.meta["categories"])}
        self.seg_model = fcn_resnet50(weights=weights, progress=False)
        self.seg_model = self.seg_model.eval()
        self.game_performance_csv = None
        self.game_manager_init()
        self.buffer.switch_on()

    def game_manager_init(self):
        self.action_request = None
        self.last_game_action = None
        self.sequence_request_topic = '/migrave/sequence_learning_game/action_request'
        self.sequence_topic = '/migrave/sequence_learning_game/action'
        self.game_performance_data_topic = '/migrave/game_performance/data'
        self.game_performance_ack_topic = '/migrave/game_performance/ack'
        self.game_performance_time_trigger_topic = '/migrave/game_performance/start_time/trigger'
        self.game_performance_time_ack_topic = '/migrave/game_performance/start_time/ack'

        self.game_performance = []
        self.acknowledging_game_performance_time = False
        self.acknowledging_game_performance = False
        self.user_play_start = None

        self.difficulty_level = 3
        self.answer_correctness = 0
        self.new_action_id = 0
        self.old_action_id = -1
        self.emotions_ids = list(range(1, 5))
        self.action = 0
        self.game_start = 0

        self.sequence_request_sub = rospy.Subscriber(self.sequence_request_topic,
                                                     Int32,
                                                     self.sequence_request_cb)

        self.sequence_pub = rospy.Publisher(self.sequence_topic, GameAction, queue_size=1, latch=True)

        self.game_performance_data_sub = rospy.Subscriber(self.game_performance_data_topic,
                                                          GamePerformance,
                                                          self.game_performance_data_cb)

        self.game_performance_ack_pub = rospy.Publisher(self.game_performance_ack_topic, Bool, queue_size=1, latch=True)

        self.game_performance_time_ack_pub = rospy.Publisher(self.game_performance_time_ack_topic, Bool, queue_size=1,
                                                             latch=True)

        self.game_performance_time_trigger_sub = rospy.Subscriber(self.game_performance_time_trigger_topic,
                                                                  Bool,
                                                                  self.game_performance_time_trigger_cb)

    def __del__(self):
        try:
            self.game_performance_csv.close()
        except Exception as e:
            repr(e)

    def game_performance_time_trigger_cb(self, trigger_msg: Bool) -> None:
        if not self.acknowledging_game_performance_time:
            self.buffer.switch_off()

            self.acknowledging_game_performance_time = True
            self.game_performance_time_ack_pub.publish(Bool(data=True))
            self.user_play_start = datetime.datetime.now()  # rospy.Time.now()
            rospy.sleep(2)
            self.game_performance_time_ack_pub.publish(Bool(data=False))
            print('Measuring time start {}'.format(self.user_play_start.strftime("%Y-%m-%d %H-%M-%S")))
            self.acknowledging_game_performance_time = False

    def game_performance_data_cb(self, performance_msg: GamePerformance) -> None:
        if not self.acknowledging_game_performance:
            self.buffer.switch_on()

            self.acknowledging_game_performance = True

            current_performance = {'solving_end_time': datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")}
            current_performance['game_start_time'] = self.game_start.strftime("%Y-%m-%d %H-%M-%S")
            current_performance['game_id'] = performance_msg.game_activity.game_id
            current_performance['action'] = self.action
            current_performance['game_activity_id'] = performance_msg.game_activity.game_activity_id
            current_performance['difficulty_level'] = performance_msg.game_activity.difficulty_level
            current_performance['answer_correctness'] = performance_msg.answer_correctness

            self.difficulty_level = current_performance['difficulty_level']
            self.answer_correctness = current_performance['answer_correctness']

            self.game_performance_ack_pub.publish(Bool(data=True))
            rospy.sleep(2)
            self.game_performance_ack_pub.publish(Bool(data=False))

            duration_time = datetime.datetime.now() - self.user_play_start
            current_performance['solving_duration'] = duration_time.total_seconds()
            self.game_performance.append(current_performance)

            self.acknowledging_game_performance = False

            self.game_performance_csv = open(os.path.join(self.output_path, f"game_performance_{self.mode}.csv"), 'a')
            labels_to_save = ['game_start_time', 'action', 'solving_end_time',
                              'game_id', 'game_activity_id', 'difficulty_level',
                              'answer_correctness', 'solving_duration']
            game_performance_writer = DictWriter(game_performance_csv, fieldnames=labels_to_save)
            game_performance_writer.writerow(current_performance)
            self.game_performance_csv.close()

    def sequence_request_cb(self, request_msg: Int32) -> None:
        self.action_request = request_msg
        self.new_action_id = request_msg.data

    def generate_random_sequence(self, min, max, length):
        sequence = list(np.random.randint(min, max + 1, size=(length)))
        while True:
            for i in range(2, len(sequence)):
                if sequence[i] == sequence[i - 1] and sequence[i - 1] == sequence[i - 2]:
                    sequence = list(np.random.randint(min, max + 1, size=(length)))
                    break
                if i == len(sequence) - 1:
                    return sequence

    def get_prediction(self, frames, activity_vector=[1, 1, 0, 0]):
        norm_frames = (np.asarray(frames, dtype=np.float32)-self.mean)/self.std
        norm_frames = torch.from_numpy(norm_frames)
        activity = torch.from_numpy(np.asarray(activity_vector, dtype=np.float32)[np.newaxis, :])
        prediction = self.model.predict((norm_frames, activity))
        return prediction

    def merge_frames(self, frames, prediction):
        image = np.concatenate([frame for frame in frames], axis=2)[0]
        image = cv2.putText(image, self.class_map[int(prediction)], (0, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,0,255), 2, cv2.LINE_AA)
        for i in range(8):
            image = cv2.rectangle(image, (i*198+56, 198), (i*198+142, 198-132), (0,0,255), 2)

        return image

    def get_activity_vector(self, difficulty_level, answer_correctness):
        x = answer_correctness
        difficulty_map = {3: [x, 1, 0, 0], 5: [x, 0, 1, 0], 7: [x, 0, 0, 1]}
        return difficulty_map[difficulty_level]

    def act(self) -> None:
        # try:
        if self.action_request is not None and self.new_action_id is not None and \
                self.old_action_id is not None:
            self.action_request = None

            print('Robot is ready to perform say the sequence')

            if self.last_game_action is None:
                game_action = GameAction()
                difficulty_level = self.difficulty_level
                answer_correctness = self.answer_correctness
            else:
                game_action = self.last_game_action
                difficulty_level = game_action.difficulty_level
                answer_correctness = game_action.answer_correctness

            if self.new_action_id != self.old_action_id:
                self.buffer.switch_off()
                frames = self.buffer.get_frames()
                self.buffer.reset()
                activity_vector = self.get_activity_vector(difficulty_level, answer_correctness)
                self.action = self.get_prediction(frames=frames, activity=activity_vector)
                merged_frames = self.merge_frames(frames)
                image_message = self.bridge.cv2_to_imgmsg(merged_frames)
                self.image_pub.publish(image_message)

                print("Selected action ", self.action)

                feedback = 0
                perform_feedback = False
                difficulty_level = 0

                if self.action == 0:
                    sequence_data = self.generate_random_sequence(self.emotions_ids[0], len(self.emotions_ids),
                                                                  length=3)
                    difficulty_level = 5
                elif self.action == 1:
                    sequence_data = self.generate_random_sequence(self.emotions_ids[0], len(self.emotions_ids),
                                                                  length=5)
                    difficulty_level = 7
                elif self.action == 3:
                    feedback = np.random.randint(low=1, high=3, size=1)[0] #can be either 1 or 2
                    perform_feedback = True
                    difficulty_level = game_action.difficulty_level
                    sequence_data = self.generate_random_sequence(self.emotions_ids[0], len(self.emotions_ids),
                                                                  length=difficulty_level)

                game_action.emotions = sequence_data
                game_action.feedback = feedback
                game_action.difficulty_level = difficulty_level
                game_action.perform_feedback = perform_feedback
                game_action.id = self.new_action_id
                self.old_action_id = self.new_action_id

            game_action.stamp = rospy.Time.now()
            self.last_game_action = game_action
            self.sequence_pub.publish(game_action)
            self.buffer.switch_on()
            self.game_start = datetime.datetime.now()

            print(
                'Time: {} \n Id: {} \n Sending the sequence: {} \n Sending feedback: {} \n Perform feedback: {}'.format(
                    datetime.datetime.fromtimestamp(game_action.stamp.to_sec()).strftime('%Y-%m-%d %H:%M:%S'),
                    game_action.id, list(game_action.emotions), game_action.feedback, game_action.perform_feedback))

        # except IndexError as err:
        #     print('No more actions in the sequence!', err)
        #     raise rospy.ROSInterruptException


if __name__ == '__main__':
    
    rospy.init_node('deep_behaviour_model')
    rospack = rospkg.RosPack()
    package_path = "/home/michal/migrave_ws/src/migrave_deep_behaviour_model" #rospack.get_path('migrave_deep_behaviour_model')

    log_path = rospy.get_param('~output_directory', "/home/qtrobot/1/game_performance/")
    model_name = rospy.get_param('~checkpoint_name', "dlc8_extended")

    deep_behaviour_model = DeepBehaviourModelWrapper(package_path=package_path,
                                                     log_path=log_path,
                                                     model_name=model_name)
    
    rospy.sleep(1.0)
    rate = rospy.Rate(1)

    try:
        while not rospy.is_shutdown():
            deep_behaviour_model.act()
            rate.sleep()
    except rospy.ROSInterruptException as exc:
        print('Deep behaviour model wrapper exiting...')
