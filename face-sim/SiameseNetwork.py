import torch
import torch.nn as nn

'''
Base Siamese Network for q1
'''


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        self.cnn = nn.Sequential(
            # Parameters: 64 features of size 5x5, stride of (1,1), padding=2
            nn.Conv2d(3, 64, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),

            # Parameters: 128 features of size 5x5, stride of (1,1), padding=2
            nn.Conv2d(64, 128, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),

            # Parameters: 256 features of size 3x3, stride of (1,1), padding=1
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),

            # Parameters: 256 features of size 3x3, stride of (1,1), padding=1
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512),
        )

        self.fc = nn.Sequential(
            nn.Linear(131072, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024)
        )

        self.fc_out = nn.Sequential(
            nn.Linear(2048, 1),
            nn.Sigmoid())

    def forward_once(self, x):
        out = self.cnn(x)
        flat_layer = out.view(out.size()[0], -1)
        return self.fc(flat_layer)

    def forward(self, img1, img2):
        f1 = self.forward_once(img1)
        f2 = self.forward_once(img2)
        f12 = torch.cat([f1, f2], 1)
        return self.fc_out(f12)
