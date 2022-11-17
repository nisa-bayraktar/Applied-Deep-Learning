import torch
from torch import nn
from torch.nn import functional as F
from imageshape import ImageShape

# Potential Improvement: Batch normalisation?
class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int):
        super().__init__()
        # Input image
        self.input_shape = ImageShape(height=height, width=width, channels=channels)

        # First convolution layer
        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(5,5),
            padding=(2, 2)
        )
        # Initialse the layer weights and bias
        self.initalise_layer(self.conv1)

        # First max pooling layer (after first convolution)
        self.max_pool1 = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(2,2)
        )

        # Second convolution layer
        self.conv2 = nn.Conv2d(
            in_channels=self.conv1.out_channels,
            out_channels=64,
            kernel_size=(3,3),
            padding=(1,1)
        )
        # Initialse the layer weights and bias
        self.initalise_layer(self.conv2)

        # Second max pooling layer (after second convolution)
        self.max_pool2 = nn.MaxPool2d(
            kernel_size=(3,3),
            stride=(2,2)
        )

        # Third convolution layer
        self.conv3 = nn.Conv2d(
            in_channels=self.conv2.out_channels,
            out_channels=128,
            kernel_size=(3,3),
            padding=(1,1)
        )
        # Initialse the layer weights and bias
        self.initalise_layer(self.conv3)

        # Third max pooling layer (after third convolution)
        self.max_pool3 = nn.MaxPool2d(
            kernel_size=(3,3),
            stride=(2,2)
        )

        # First fully conntected layer
        self.full_connect1 = nn.Linear(15488, 4608)
        self.initalise_layer(self.full_connect1)

        # Second fully connected layer
        self.full_connect2 = nn.Linear(2304, 2304)
        self.initalise_layer(self.full_connect2)

    #computes the forward pass through all network layers
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.conv1(images)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.max_pool3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.full_connect1(x)
        #split and maxout layer
        x = x.reshape(x.size(0), 2304, 2)
        x = torch.max(x, dim=2)[0]
        #final fully connected layer
        x = self.full_connect2(x)
        return x

    @staticmethod
    def initalise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.constant_(layer.bias, 0.1)
        if hasattr(layer, "weight"):
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
