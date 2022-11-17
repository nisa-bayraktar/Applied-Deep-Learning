import time
from multiprocessing import cpu_count
from typing import Union
import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import pickle

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        summary_writer: SummaryWriter,
        device: torch.device,
    ):
    #initialise trainer parameters
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.summary_writer = summary_writer
        self.step = 0

    def train(self, lrs, checkpoint_path, checkpoint_frequency: int, epochs: int, val_frequency: int, log_frequency: int = 5, start_epoch: int = 0):
        #put model into training mode
        self.model.train()
        for epoch in range(start_epoch, epochs):
            #put model into training mode
            self.model.train()
            data_load_start_time = time.time()
            #update learning rate for current epoch
            for group in self.optimizer.param_groups:
                group['lr'] = lrs[epoch]

            for batch, labels in self.train_loader:
                #convert batch labels to CUDA
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                data_load_end_time = time.time()
                #compute forward pass
                logits = self.model.forward(batch)
                #compute loss
                loss = self.criterion(logits, labels)
                #compute gradient
                loss.backward()

                #perform weight update
                self.optimizer.step()
                #reset gradient to 0
                self.optimizer.zero_grad()

                #log values: save to file and print to terminal
                data_load_time = data_load_end_time - data_load_start_time
                step_time = time.time() - data_load_end_time
                if ((self.step + 1) % log_frequency) == 0:
                    self.log_metrics(epoch, loss, data_load_time, step_time)
                if ((self.step + 1) % 2) == 0:
                    self.print_metrics(epoch, loss, data_load_time, step_time)

                #increase step
                self.step += 1
                data_load_start_time = time.time()
            
            #save model as checkpoint at appropriate frequency
            if (epoch + 1) % checkpoint_frequency or (epoch + 1) == epochs:
                print(f"Saving model to {checkpoint_path}")
                torch.save(self.model.state_dict(), checkpoint_path)

            self.summary_writer.add_scalar("epoch", epoch, self.step)
            #validate model at given frequency
            if ((epoch + 1) % val_frequency) == 0:
                #validate model, but dont save outputs to file
                self.validate(False)
                #put model back into training mode
                self.model.train()
        #do final model validation, and save predictions to file
        self.validate(True)

    #add log metrics to log file
    def log_metrics(self, epoch, loss, data_load_time, step_time):
        self.summary_writer.add_scalar("epoch",epoch,self.step)
        self.summary_writer.add_scalars("loss",{"train": float(loss.item())},self.step)
        self.summary_writer.add_scalar("time/data",data_load_time,self.step)
        self.summary_writer.add_scalar("time/data",step_time,self.step)

    #print appropriate information
    def print_metrics(self, epoch, loss, data_load_time, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"data load time: "
                f"{data_load_time:.5f}, "
                f"step time: {step_time:.5f}"
        )

    #validation of model
    def validate(self, output):
        preds = []
        total_loss = 0
        #put model into evaluation mode
        self.model.eval()
        with torch.no_grad():
            for batch, labels in self.val_loader:
                #convert to CUDA
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                #compute forward pass
                logits = self.model(batch)
                #compute loss
                loss = self.criterion(logits, labels)
                #get total loss
                total_loss += loss.item()
                #stored prediction
                preds.extend(logits.cpu().numpy())

        #if function input is true, save predictions to file
        if(output == True):
            output_file = open('preds.pkl', 'wb')
            pickle.dump(preds, output_file)
            output_file.close()

        #calculate average loss
        average_loss = total_loss / len(self.val_loader)
        self.summary_writer.add_scalars("loss",{"test": average_loss},self.step)
        print(f"validation loss: {average_loss:.5f}")
