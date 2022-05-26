import torch.nn as nn 
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, in_channels, n_classes=10):
        """
        Initialize LeNet classifier

        Inputs:
        in_channels : int
            Number of input channels
        n_classes : int, default=10
            Number of classes
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=n_classes)

    def forward(self, x):
        """
        Implement classification using LeNet

        Inputs:
        x : torch.tensor of shape (batch_size, channels, width, height)
            Input data

        Outputs:
        out : torch.tensor of shape (n_classes)
            Unnormalized output
        """
        feat_dims = dict()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        feat_dims['pre-flattened'] = x.shape
        x = F.torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out

class BaselineLeNet(LeNet):
    def __init__(self, in_channels, n_classes=10):
        LeNet.__init__(self, in_channels, n_classes=n_classes)

    def forward(self, x):
        """
        Implement classification using LeNet

        Inputs:
        x : torch.tensor of shape (batch_size, channels, width, height)
            Input data

        Outputs:
        prob_out : torch.tensor of shape (n_classes) 
            Probability output
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        prob_out = F.softmax(out, dim=1)
        return prob_out