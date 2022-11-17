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

torch.backends.cudnn.benchmark = True

#create input parser
parser = argparse.ArgumentParser(
    description="Train CNN on SALICON",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

#add parser arguments
parser.add_argument("--log-dir", default=Path("logs"), type=Path)
parser.add_argument("--learning-rate", default=0.03, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=128, type=int, help="Number of images within each mini-batch")
parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs")
parser.add_argument("--val-frequency", default=5, type=int, help="How frequently to test the model on the validation set")
parser.add_argument("--log-frequency",default=5,type=int,help="How frequently to save logs to tensorboard in number of steps")
parser.add_argument("-j", "--worker-count", default=cpu_count(), type=int, help="Number of worker processes")

parser.add_argument("--checkpoint-path", type=Path, default="checkpoints.pt")
parser.add_argument("--checkpoint-frequency", type=int, default=50, help="Save a checkpoint every N epochs")
parser.add_argument("--resume-checkpoint", type=Path)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

def main(args):
    train_dataset = Salicon("train.pkl")
    val_dataset = Salicon("val.pkl")

    #load datasets
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count
    )

    #initialise model with correct size
    model = CNN(height=96, width=96, channels=3)
    #set loss function
    criterion = nn.L1Loss()
    #criterion = nn.MSELoss() //Old loss function

    #set optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=0.0005, nesterov=True)
    #initialise learning rates
    learning_rates = np.linspace(args.learning_rate, 0.0001, args.epochs)

    #load model from checkpoint if needed
    if args.resume_checkpoint is not None:
        state_dict = torch.load(args.resume_checkpoint)
        print(f"Loading model from {args.resume_checkpoint}")
        model.load_state_dict(state_dict)

    #initialise the log directory
    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(str(log_dir), flush_secs=5)
    #initialise trainer class
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, summary_writer, DEVICE)
    #start training the model
    trainer.train(learning_rates, args.checkpoint_path, args.checkpoint_frequency, args.epochs, args.val_frequency, log_frequency=args.log_frequency)
    summary_writer.close()

def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    tb_log_dir_prefix = (f"CNN_bn"
                         f"_bs={args.batch_size}_"
                         f"lr={args.learning_rate}"
                         f"_run_"
                         )
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)

if __name__ == "__main__":
    main(parser.parse_args())
