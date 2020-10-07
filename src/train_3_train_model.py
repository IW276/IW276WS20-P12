import argparse
import json
import copy
import logging
import os
import time
import numpy as np

from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from ddnet.data_loader import DDNetDataset
from ddnet.ddnet import DDNet, DDNetConfig

from tqdm import tqdm

DATASETS_DIR = '../datasets/'
MODELS_DIR = "../pretrained-models/"
TRAINING_DATA_DIR = os.path.join(DATASETS_DIR, "training-data")

CLASS_NAMES = {
        "walking, general" : 1,
        "walking the dog" : 2,
        "running" : 3,
        "jogging" : 4,
        "bicycling, general" : 5
}

LOG_PER_N_ITER = 100

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default=MODELS_DIR)
parser.add_argument('--train_dir', type=str, default=TRAINING_DATA_DIR)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_input_frames', type=int, default=6, help='number of frames for attention window')
parser.add_argument('--n_epochs', type=int, default=50)
parser.add_argument('--extra_log_dir', type=str)
parser.add_argument('--gpu', type=int, default=0)
cfg = parser.parse_args()

TRAINING_IDENTIFIER = f'b{cfg.batch_size}_gpu{cfg.gpu}_epochs{cfg.n_epochs}_inp{cfg.n_input_frames}'


def train(model, gpu, model_dir, data_loaders, criterion, optimizer, num_epochs=25, lr_scheduler=None, extra_log_dir=None):
    """
    https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    """
    tmp_name = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    logging_fn = f'{tmp_name}_{TRAINING_IDENTIFIER}.log'
    #model_dir = f'{model_dir}/{tmp_name}_{TRAINING_IDENTIFIER}'
    model_dir = os.path.join(cfg.model_dir, "{}_{}".format(tmp_name, TRAINING_IDENTIFIER))
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    # Log to both file & console: https://stackoverflow.com/a/46098711
    log_handlers = [logging.StreamHandler(), logging.FileHandler(os.path.join(model_dir, logging_fn))]
    if extra_log_dir is not None:
        log_handlers.append(logging.FileHandler(os.path.join(extra_log_dir, logging_fn)))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=log_handlers)

    since = datetime.now()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    phase_log_text = {'train': '_', 'val': '_'}
    for epoch in tqdm(range(num_epochs)):
        logging.info(f'Epoch {epoch}/{num_epochs - 1}')
        logging.info('-' * 10)

        # Each epoch has a training and validation phase
        eval_pred = np.empty((0, 19))
        eval_true = np.empty(0)
        for phase in ['train', 'val']:
            since_epoch = datetime.now()
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for i_batch, batch in tqdm(enumerate(data_loaders[phase])):
                batch = [x.to(f'cuda:{gpu}') for x in batch]
                if model.name == 'ddnet':
                    *inputs, labels = batch
                    inputs = [x.float().permute((0, 2, 1)) for x in inputs]  # Reshape BLC (Keras) to BCL (PyTorch)
                    labels = labels.long()
                else:
                    inputs, labels = batch
                batch_size = len(labels)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    if model.name == 'ddnet':
                        outputs = model(*inputs)
                    else:
                        outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    if phase == 'val':
                        np_outputs = outputs.to('cpu')
                        np_labels = labels.to('cpu')
                        eval_pred = np.vstack((eval_pred, np_outputs.numpy()))
                        eval_true = np.concatenate((eval_true, np_labels.numpy()))

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)

                if phase == 'train' and i_batch % LOG_PER_N_ITER == 0:
                    cum_loss = running_loss / ((i_batch + 1) * batch_size)
                    cum_acc = running_corrects.double() / ((i_batch + 1) * batch_size)
                    n_batches = round(len(data_loaders[phase].dataset) / batch_size + 0.5)
                    logging.info(f'{phase} epoch {epoch} batch {i_batch}/{n_batches} '
                                 f'Loss: {cum_loss:.4f} Acc: {cum_acc:.4f}')

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(data_loaders[phase].dataset)

            logging.info(f'{phase} after epoch {epoch} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model, os.path.join(model_dir, f'current_best.pth'))
                logging.info(f"New best model at epoch {epoch:02d} (saved in current_best.pth)")
            if phase == 'val':
                val_acc_history.append(epoch_acc)

            # Save checkpoint
            torch.save(model, os.path.join(model_dir, f'Epoch{epoch:02d}.pth'))

            # LR scheduler at end of val
            if lr_scheduler is not None and phase == 'val':
                lr_scheduler.step(epoch_loss)

            logging.info('-' * 10)
            epoch_time = datetime.now() - since_epoch
            phase_log_text[phase] = f'Phase: {phase.upper()}\nduration: {epoch_time}\nloss: {epoch_loss:.4f}\nacc: {epoch_acc:.4f}'

    time_elapsed = datetime.now() - since
    logging.info(f'Training completed after {time_elapsed}')
    logging.info(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    # Save best model
    torch.save(model, os.path.join(model_dir, f'Best.pth'))

    return model, val_acc_history


if __name__ == '__main__':
    #for train_file in os.listdir(TRAINING_DATA_DIR):
    #    with open(os.path.join(cfg.train_dir, train_file)) as train_json:
    #        train_data = json.loads(train_json.read())
    #    train_set = DDNetDataset(train_data, 'train', cfg.n_input_frames)
    #    data_loaders = dict()
    #    data_loaders['train'] = DataLoader(train_set, cfg.batch_size, shuffle=True, num_workers=0)
    #    print(len(data_loaders['train']))
    #    # Model
    #    ddnet_cfg = DDNetConfig(len(CLASS_NAMES), cfg.n_input_frames, train_set.n_joints, train_set.d_joints)
    #    model = DDNet(ddnet_cfg)
    #    model.to(f'cuda:{cfg.gpu}')
    #    # Loss
    #    criterion = nn.CrossEntropyLoss()
    #    # Optimizer
    #    optimizer = optim.Adam(model.parameters())
    #    # LR scheduler
    #    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    #    # Train
    #    train(model, cfg.gpu, cfg.model_dir, data_loaders, criterion, optimizer, cfg.n_epochs, lr_scheduler, cfg.extra_log_dir)
    train_set = DDNetDataset(cfg.train_dir, 'train', cfg.n_input_frames)
    data_loaders = dict()
    data_loaders['train'] = DataLoader(train_set, cfg.batch_size, shuffle=True, num_workers=0)
    # Model

    ddnet_cfg = DDNetConfig(len(CLASS_NAMES), cfg.n_input_frames, train_set.n_joints, train_set.d_joints)

    model = DDNet(ddnet_cfg)
    model.to(f'cuda:{cfg.gpu}')
    # Loss
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.Adam(model.parameters())
    # LR scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    # Train
    train(model, cfg.gpu, cfg.model_dir, data_loaders, criterion, optimizer, cfg.n_epochs, lr_scheduler, cfg.extra_log_dir)
