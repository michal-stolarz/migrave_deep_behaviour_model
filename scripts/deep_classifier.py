import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from network import DLC
import config_dlc as dcfg
import re
import matplotlib.pyplot as plt

torch.manual_seed(42)


class DeepClassifier:
    def __init__(self, path, cfg=dcfg, validation=False, input_state_size=1):
        # cpu or cuda
        self.device = cfg.device  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = cfg.proc_frame_size  # State dimensionality 84x84.
        self.state_size_data = cfg.state_size_data
        self.minibatch_size = cfg.minibatch_size
        self.learning_rate = cfg.learning_rate
        self.validation = validation
        self.epochs_num = cfg.epochs
        self.epoch_end = self.epochs_num
        self.epoch_start = 0
        self.input_state_size = input_state_size
        self.classes_num = cfg.noutputs
        self.conf_matrix_labels = cfg.labels
        self.predictions = None

        if self.input_state_size == 1:
            nstates = cfg.nstates
            nfeats = cfg.nfeats
            checkpoint_path = os.path.join(path, 'checkpoint/dlc1')

        elif self.input_state_size == 8:
            nstates = cfg.nstates_full
            nfeats = cfg.nfeats_full
            checkpoint_path = os.path.join(path, 'checkpoint/dlc8')
            
        else:
            raise ValueError("Unaccountable state size")

        self.model = DLC(noutputs=cfg.noutputs, nfeats=nfeats,
                         nstates=nstates, kernels=cfg.kernels,
                         strides=cfg.strides, poolsize=cfg.poolsize,
                         enable_activity_signals=True)

        checkpoint_name = 'model_epoch'
        pattern = re.compile(checkpoint_name + '[0-9]+.pt')

        # metrics
        #self.accuracy = torchmetrics.classification.Accuracy(task='multiclass', average='macro', num_classes=self.classes_num)
        #self.precision = torchmetrics.classification.Precision(task="multiclass", average='macro', num_classes=self.classes_num)
        #self.recall = torchmetrics.classification.Recall(task="multiclass", average='macro', num_classes=self.classes_num)
        #self.f1_score = torchmetrics.classification.F1Score(task="multiclass", average='macro', num_classes=self.classes_num)
        #self.conf_mat = torchmetrics.classification.ConfusionMatrix(task="multiclass", num_classes=self.classes_num)

        params_count = self.model.get_params_count()
        print(f"Number of trainable parameters {params_count}")

        #self.optimizer = optim.Adam(self.model.parameters(), self.learning_rate)
        #self.loss_fuction = nn.CrossEntropyLoss()

        try:
            if os.path.exists(checkpoint_path):
                entries = sorted(os.listdir(checkpoint_path), reverse=True)
                if entries:
                    numbers = [int(re.findall('\d+', entry)[0]) for entry in entries if pattern.match(entry)]
                    epoch = max(numbers)

                    self.epoch_end = epoch + self.epochs_num + 1
                    self.epoch_start = epoch + 1

                    checkpoint = torch.load(os.path.join(checkpoint_path, f"{checkpoint_name}{epoch}.pt"), map_location=torch.device('cpu'))
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print("MODEL FOUND")
                else:
                    print("NO MODEL FOUND")
            else:
                print("NO MODEL FOUND")

        except Exception as err:
            print(repr(err))

    def predict(self, tensor):
        self.model.eval()
        images, activity = tensor

        images, activity = images.to(self.device), activity.to(self.device)
        full_state = [images, activity]

        with torch.no_grad():
            action_values = self.model(full_state)

        return torch.argmax(action_values.cpu().detach().clone(), dim=1)

    def majority_vote(self, tensor):
        images, activity = tensor

        if self.input_state_size != 1:
            raise 'Function unusable with input state size != 1'

        images = torch.unsqueeze(images[0], dim=1)
        activity = activity.repeat(images.shape[0], 1)
        full_state = [images, activity]
        predictions = self.predict(full_state)
        self.predictions = predictions

        return np.bincount(predictions).argmax()
