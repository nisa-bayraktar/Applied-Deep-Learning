import torch
from torch import nn
from torch.nn import functional as F
from audioshape import AudioShape

# Potential Improvement: Batch normalisation?
class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int):
        super().__init__()
        # Input audio
        self.input_shape = AudioShape(height=height, width=width, channels=channels)

        # First convolution layer
        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=16,
            kernel_size=(10,23),
            padding='same'
        )
        # Initialse the layer weights and bias
        self.initalise_layer(self.conv1)

        # First max pooling layer (after first convolution)
        self.max_pool1 = nn.MaxPool2d(
            kernel_size=(1, 20),
            stride=(2,2)
        )

        self.conv1b = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=16,
            kernel_size=(21,20),
            padding='same'
        )
        self.max_pool1b = nn.MaxPool2d(
            kernel_size=(20, 21),
            stride=(2,2)
        )

        # First fully connected layer
        self.full_connect1 = nn.Linear(320, 5120)
        self.initalise_layer(self.full_connect1)

        # Second fully connected layer
        # self.full_connect2 = nn.Linear(2304, 2304)
        # self.initalise_layer(self.full_connect2)

    #computes the forward pass through all network layers
    def forward(self, audios: torch.Tensor) -> torch.Tensor:
        x = self.conv1(audios)
        x = F.leaky_relu(x, 0.3)
        x = self.max_pool1(x)
        # second convolution pipeline
        xb = self.conv1b(audios)
        xbRelu = F.leaky_relu(0.3)
        xb = F.leaky_relu(xb,0.3)
        xb = self.max_pool2(xb)
        x = torch.flatten(x, start_dim=1)
        xb = torch.flatten(xb,start_dim=1)
        print(x.shape())
        print(xb.shape())
        # merge the two output
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

 model = CNN(height=80, width=80, channels=1) 
 model.forward(dataset)
