import numpy as np
import torch
import os
from network import DLC
import config_dlc as dcfg

torch.manual_seed(42)


class DeepClassifier:
    def __init__(self, path, model_name, cfg=dcfg, validation=False, input_state_size=1, input_modalities=1):
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
        self.input_modalities = input_modalities
        self.classes_num = cfg.noutputs
        self.conf_matrix_labels = cfg.labels
        self.predictions = None

        nstates = cfg.nstates_full
        nfeats = cfg.nfeats_full
        checkpoint_path = os.path.join(path, 'checkpoint')

        self.model = DLC(noutputs=cfg.noutputs,
                         nfeats=nfeats,
                         nstates=nstates,
                         kernels=cfg.kernels,
                         strides=cfg.strides,
                         poolsize=cfg.poolsize,
                         enable_activity_signals=True)

        params_count = self.model.get_params_count()
        print(f"Number of trainable parameters {params_count}")

        try:
            if os.path.exists(checkpoint_path):
                entry = sorted(os.listdir(checkpoint_path), reverse=True)[0]

                checkpoint = torch.load(os.path.join(checkpoint_path, entry),
                                        map_location=torch.device('cpu'))
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print("MODEL FOUND")
            else:
                print("NO MODEL FOUND")

        except Exception as err:
            print(repr(err))

    def predict(self, tensor):
        self.model.eval()

        if self.input_modalities == 2:
            images, activity = tensor
            images, activity = images.to(self.device), activity.to(self.device)
            images = torch.movedim(images, 0, 1)
            full_state = [images, activity]
        else:
            images = tensor
            images = images.to(self.device)
            full_state = images

        with torch.no_grad():
            action_values = self.model(full_state)

        return torch.argmax(action_values.cpu().detach().clone(), dim=1)

    def majority_vote(self, tensor):
        if self.input_modalities == 2:
            images, activity = tensor
        else:
            images = tensor

        if self.input_state_size != 1:
            raise 'Function unusable with input state size != 1'

        images = torch.unsqueeze(images[0], dim=1)

        if self.input_modalities == 2:
            activity = activity.repeat(images.shape[0], 1)
            full_state = [images, activity]
        else:
            full_state = images

        predictions = self.predict(full_state)
        self.predictions = predictions

        return np.bincount(predictions).argmax()
