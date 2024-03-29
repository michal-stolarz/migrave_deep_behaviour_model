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
        classifier = nn.Sequential(
            nn.Linear(self.nstates[2] * self.kernels[1] * self.kernels[1], self.nstates[3]),
            nn.ReLU(),
            nn.Linear(self.nstates[3], self.noutputs),
        )
        
        if self.enable_activity_signals:
            self.linear = nn.Sequential(
                nn.Linear(self.activity_vector_size, 16),
                nn.ReLU()
            )

            self.fc1 = nn.Sequential(
                nn.Linear(self.nstates[2] * self.kernels[1] * self.kernels[1], 16),
                nn.ReLU()
            )
            self.fc2 = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU()
            )
            self.classifier = nn.Sequential(
                nn.Linear(16, self.noutputs),
            )
            
            self.image_classifier = classifier
          
        else:
            self.classifier = classifier

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
            result = self.classifier(features)

        return result

    def get_params_count(self):
        return sum(p.numel() for p in self.parameters())
