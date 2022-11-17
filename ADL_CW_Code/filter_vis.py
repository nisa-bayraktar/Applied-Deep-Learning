from cnn import CNN
from trainer import Trainer
from imageshape import ImageShape
from dataset import Salicon
from multiprocessing import cpu_count

import torch
from torch import nn
from evaluation import auc_borji
from torch.utils.tensorboard import SummaryWriter
import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def main() :
    #load the saved model from checkpoint.pt file
    model = CNN(height=96, width=96, channels=3)
    model.load_state_dict(torch.load("checkpoints.pt", map_location=torch.device('cpu')))
    #put model into evaluation mode
    model.eval()
    #visualise filters
    plot_filter(model)


def plot_filter(model) :
    #get weights after first convolutional layer
    weight_tensor = model.conv1.weight
    #get the number of filters (there are 32)
    num_kernels = weight_tensor.shape[0]
    
    #create figure to plot onto
    fig = plt.figure(figsize=(6,6))

    #for each filter
    for i in range(num_kernels):
        #add to subplot
        ax1 = fig.add_subplot(6,6,i+1)

        #normalise the values
        npImg = np.array(weight_tensor[i].detach().numpy(), np.float32)
        npImg = (npImg - np.min(npImg)) / (np.max(npImg) - np.min(npImg))
        npImg = np.transpose(npImg)

        ax1.imshow(npImg)
        ax1.axis('off')
        
    #save to file
    plt.savefig('filters.png', dpi=200)    
    plt.close()


if __name__ == '__main__'  :
    main()