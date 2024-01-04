import torch
import torch.nn as nn

class DLC(nn.Module):
    def __init__(self, noutputs, nfeats, nstates, kernels, strides, poolsize, enable_activity_signals=False,
                 activity_vector_size=4):
        super(DLC, self).__init__()
        self.noutputs = noutputs
        self.nfeats = nfeats
        self.nstates = nstates
        self.kernels = kernels
        self.strides = strides
        self.poolsize = poolsize
        self.enable_activity_signals = enable_activity_signals
        self.activity_vector_size = activity_vector_size
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=self.nfeats, out_channels=self.nstates[0], kernel_size=self.kernels[0],
                      stride=self.strides[0], padding=1),
            nn.BatchNorm2d(nstates[0]),
            nn.ReLU(),
            nn.MaxPool2d(self.poolsize),
            nn.Conv2d(in_channels=self.nstates[0], out_channels=self.nstates[1], kernel_size=self.kernels[1],
                      stride=self.strides[1]),
            nn.BatchNorm2d(self.nstates[1]),
            nn.ReLU(),
            nn.MaxPool2d(self.poolsize),
            nn.Conv2d(in_channels=self.nstates[1], out_channels=self.nstates[2], kernel_size=self.kernels[1],
                      stride=self.strides[1]),
            nn.BatchNorm2d(self.nstates[2]),
            nn.ReLU(),
            nn.MaxPool2d(self.poolsize),
        )
        self.image_classifier = nn.Sequential(
            nn.Linear(self.nstates[2] * self.kernels[1] * self.kernels[1], self.nstates[3]),
            nn.ReLU(),
            nn.Linear(self.nstates[3], self.noutputs)
        )

        self.linear = nn.Sequential(
            nn.Linear(self.activity_vector_size, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(self.nstates[2] * self.kernels[1] * self.kernels[1], self.nstates[3]),
            nn.ReLU(),
            nn.Linear(self.nstates[3], 2)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(8, self.noutputs),
        )

    def forward(self, x):
        if self.enable_activity_signals and len(x)!=2:
            raise NotImplementedError

        if self.enable_activity_signals:
            image = x[0]
            activity_vector = x[1]
        else:
            image = x

        features = self.features(image)
        features = features.view(features.size(0), self.nstates[2] * self.kernels[1] * self.kernels[1])

        if self.enable_activity_signals:
            conv = self.fc1(features)
            ac = self.linear(activity_vector)
            cat = torch.cat((conv, ac), 1)
            cat = self.fc2(cat)
            result = self.classifier(cat)
        else:
            result = self.image_classifier(features)

        return result

    def get_params_count(self):
        return sum(p.numel() for p in self.parameters())