import torch
from torch import nn
from torch.nn import functional as F
from audioshape import AudioShape

# Potential Improvement: Batch normalisation?
class CNN(nn.Module):
    def __init__(self, height: int, width: int, channels: int, class_count=int):
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
        )

        self.conv1b = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=16,
            kernel_size=(21,20),
            padding='same'
        )

        self.initalise_layer(self.conv1b)

        self.max_pool1b = nn.MaxPool2d(
            kernel_size=(20, 1),
        )

        # First fully connected layer
        self.full_connect1 = nn.Linear(10240, 200)
        self.initalise_layer(self.full_connect1)

        # Second fully connected layer
        self.full_connect2 = nn.Linear(200, 10)
        self.initalise_layer(self.full_connect2)

        # dropout
        self.dropout = nn.Dropout(0.1)

        # softmax


    #computes the forward pass through all network layers
    def forward(self, audios: torch.Tensor) -> torch.Tensor:
        x = self.conv1(audios)
        print("Shape of x initially",x.shape)
        x = F.leaky_relu(x, 0.3)
        x = self.max_pool1(x)
        print("After pooling",x.shape)
        # second convolution pipeline
        xb = self.conv1b(audios)
        xb = F.leaky_relu(xb,0.3)
        xb = self.max_pool1b(xb)
        print("After pooling",xb.shape)
        x = torch.flatten(x, start_dim=1)
        xb = torch.flatten(xb,start_dim=1)
        print("After flattening",x.shape)
        print("After flattening",xb.shape)
        # merge the two output
        x = torch.cat((x,xb),1)
        print("The merged array is", x.shape)
        #fully connected layer
        x = self.full_connect1(x)
        x = F.leaky_relu(x,0.3)
        print("The size is", x.shape)

        # #final fully connected layer
        x = self.full_connect2(x)

        # dropout
        x = self.dropout(x)
        print(x)
        print("The sum is",sum(x))

        # softmax
        x = F.softmax(x)
        print("After softmax",x)
        print("The sum is",sum(x))
        print("The size is", x.shape)
        return x

    @staticmethod
    def initalise_layer(layer):
        if hasattr(layer, "bias"):
            nn.init.constant_(layer.bias, 0.1)
        if hasattr(layer, "weight"):
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)