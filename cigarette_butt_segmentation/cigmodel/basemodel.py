import numpy as np
import torch
import random
import torchvision.datasets
import torch.nn as nn
import os
from torch.optim.lr_scheduler import StepLR
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
from lib import *


class Basenet(nn.Module):
    """Class for neural net
    (a, b)Two times:
        Two convolution layers, saving the size of input (not the number of channels)
        Batch normalization
        MaxPool 4x4 with stride 4
    (c)One time:
        Two convolution layers, saving the size of input
        Batch normalization
        MaxPool 2x2 with stride 2
    (d)Two convolution layers, saving the size of input

    Transposed convolution, increasing the size twice

    (e, f)Two times:
        Two convolution layers, saving the size of input
        Batch normalization
        Transposed convolution, quadrupling size

    (g)Three convolution layers, saving the size of input
    The first and the second have batch normalization

    Skip connections between layers:
        (a)[2] ---> (g)[1]
        (b)[2] ---> (f)[1]
        (c)[2] ---> (e)[1

    """

    # Contract path
    def __init__(self):
        super(Basenet, self).__init__()
        self.conconv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.conconv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.conconv3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(in_channels=8, out_channels=32,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conconv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conconv5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Conv2d(in_channels=32, out_channels=128,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conconv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conconv7 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=128,  # Expanding path
                               kernel_size=2, stride=2)
        )

        self.extconv1 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=128, out_channels=32,
                               kernel_size=4, stride=4)
        )

        self.extconv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=8,
                               kernel_size=4, stride=4)
        )

        self.extconv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=8,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=1,
                      kernel_size=1)
        )

    def forward(self, x):
        x = self.conconv1(x)
        sk1 = self.conconv2(x)  # For skip-connection
        x = self.conconv2(x)

        x = self.conconv3(x)
        sk2 = self.conconv4(x)
        x = self.conconv4(x)

        x = self.conconv5(x)
        sk3 = self.conconv6(x)
        x = self.conconv6(x)
        x = self.conconv7(x)

        x = torch.cat((sk3, x), 1)
        x = self.extconv1(x)

        x = torch.cat((sk2, x), 1)
        x = self.extconv2(x)

        x = torch.cat((sk1, x), 1)
        x = self.extconv3(x)

        return x


def train_net(net, max_epoch, metric, threshold, train_batch_gen,
              val_batch_gen, optimizer, path, device, draw_history=True, scheduler=False):
    """Trains and validate the net.

    Parameters
    ----------
    net : object of class(nn.Module)
        Net to train and validate.
    max_epoch : int
        Max number of epoch to train.
    metric : function
        Metric to evaluate the model quality.
    threshold : float from 0 to 1
        Every predicted pixel with value > threshold corresponds to the cigarette.
    train_batch_gen : object of class torch.utils.data.DataLoader
        train batch generator.
    val_batch_gen : object of class torch.utils.data.DataLoader
        validation batch generator.
    optimizer
    path : str
        Path to save the model with best metric score.
    device : cpu or gpu
    draw_history : bool
     If true, draws the history of loss and metric value at the end of training and validating.
    scheduler : bool
        If true, uses learning rate scheduler.

    Returns
    -------
    train_loss_hist : list
        History of loss by each epoch while training.
    train_dice_hist : list
        History of dice metric by each epoch while training.
    val_loss_hist : list
        History of loss by each epoch while validating.
    val_dice_hist : list
        History of dice metric by each epoch while validating.

    """
    train_batch_num = len(train_batch_gen)
    val_batch_num = len(val_batch_gen)
    train_loss_hist = []
    val_loss_hist = []
    train_dice_hist = []
    val_dice_hist = []
    val_d_best = 0
    if scheduler:
        scheduler_lr = StepLR(optimizer, step_size=15, gamma=0.2)
    for epoch in range(max_epoch):
        train_loss = 0
        train_dice = 0
        val_loss = 0
        val_dice = 0
        net.train()
        for batch in train_batch_gen:
            optimizer.zero_grad()

            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            weights = batch['weight'].to(device)

            pred = net.forward(imgs)
            loss_val = weighted_loss(pred, masks, weights)
            train_loss += loss_val.data.cpu()
            train_dice += metric(masks, pred, threshold)

            loss_val.backward()
            optimizer.step()

        net.eval()
        for batch in val_batch_gen:
            imgs = batch['image'].to(device)
            masks = batch['mask'].to(device)
            weights = batch['weight'].to(device)

            pred = net.forward(imgs)
            loss_val = weighted_loss(pred, masks, weights)
            val_loss += loss_val.data.cpu()
            val_dice += metric(masks, pred, threshold)

        train_l = train_loss / train_batch_num
        val_l = val_loss / val_batch_num
        train_d = train_dice / train_batch_num
        val_d = val_dice / val_batch_num

        train_loss_hist.append(train_l)
        val_loss_hist.append(val_l)
        train_dice_hist.append(train_d)
        val_dice_hist.append(val_d)

        if val_d > val_d_best:
            val_d_best = val_d
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'values': {'train_loss': train_l,
                           'train_dice': train_d,
                           'val_loss': val_l,
                           'val_dice': val_d}
            }, path)

        print(f"Epoch number {epoch + 1}")
        print(f"Train loss = {train_l}\tTrain dice = {train_d}")
        print(f"Val loss = {val_l}\tVal dice = {val_d}")

        if scheduler:
            scheduler_lr.step()

    if draw_history:
        draw_hist(train_loss_hist, train_dice_hist,
                  val_loss_hist, val_dice_hist)

    return train_loss_hist, train_dice_hist, val_loss_hist, val_dice_hist


def dice_metric(true, pred, threshold):
    """Computes the dice metric on true and predicted sample with threshold

    Parameters
    ----------
    true : tensor, tensor.shape = [N, 512, 512]
        True masks of the cigarettes.
    pred : tensor, tensor.shape = [N, 1, 512, 512]
        Predicted masks.
    threshold : float from 0 to 1

    Returns
    -------
    float from 0 to 1
        If N (number of samples in batch) != 1, compute mean dice metric.

    """
    true = list(true.cpu().numpy())
    pred = list(np.squeeze(pred.data.cpu().numpy(), axis=1) > threshold)

    return get_dice(true, pred)


def weighted_loss(pred, masks, weights):
    """Weighted loss function.

    Parameters
    ----------
    pred : tensor, tensor.shape = [N, 1, 512, 512]
        Predicted masks.
    masks : tensor, tensor.shape = [N, 1, 512, 512]
        True masks.
    weights : tensor, tensor.shape = [N, 1, 512, 512]
        Weights for masks

    Notes
    -----
    N is the number of samples in the batch

    """
    s = nn.Sigmoid()
    batch_size = pred.shape[0]

    loss = (-1) * torch.sum(
        weights * (masks * torch.log(s(pred) + 1e-10))
        + (1 - masks) * torch.log(1 - s(pred) + 1e-10)) / batch_size

    return loss


def draw_sample(net, device, path, threshold=0.5, rand=True, test=False):
    """Draw a sample, true mask (if test is False), predicred mask and predicted mask
    with threshold

    Parameters
    ----------
    net : object of class(nn.Module)
    device : cpu or gpu
    path : str
        Path to the folder with the folder 'images'.
    threshold : float from 0 to 1
    rand : bool
        If True, shows the random sample.
    test : bool
        If True, does not use true masks.

    """
    net.eval()
    img_names = sorted(os.listdir(f"{path}/images"))
    trans_to_tens = transforms.ToTensor()
    trans_to_norm = transforms.Normalize([0.5315, 0.5226, 0.4595],
                                         [0.2005, 0.2006, 0.1882])
    s = nn.Sigmoid()
    mask = None

    if rand is True:
        idx = np.random.randint(0, len(img_names))
    else:
        idx = rand

    img = Image.open(f"{path}/images/{img_names[idx]}").convert('RGB')
    sample = trans_to_norm(trans_to_tens(img).to(device)).unsqueeze(0)
    pred = s(net.forward(sample).data.cpu())
    pred = np.squeeze(pred.numpy(), axis=(0, 1))
    pred_thr = pred > threshold

    if test is False:
        mask_names = sorted(os.listdir(f"{path}/masks"))
        mask = Image.open(f"{path}/masks/{mask_names[idx]}")
        true = np.squeeze(trans_to_tens(mask).numpy(), axis=0)
        print(f"Dice is {get_dice(true, pred_thr)}")

    draw_pics(img, mask, pred, pred_thr)


def draw_pics(img, mask, pred, pred_thr):
    """Draw image, mask (if it is not None), predicted mask and
    predicted mask with threshold.

    Parameters
    ----------
    img : PIL Image format
    mask : PIL Image format or None
    pred : 2d np.array
    pred_thr : 2d np.array

    """
    if mask is not None:
        _, ax = plt.subplots(1, 4, figsize=(15, 15))
        ax[0].imshow(img)
        ax[0].set_title('Real image')
        ax[1].imshow(mask)
        ax[1].set_title('Mask for cigarette')
        ax[2].imshow(pred)
        ax[2].set_title('Predicted mask')
        ax[3].imshow(pred_thr)
        ax[3].set_title('Predicted mask with threshold')
    else:
        _, ax = plt.subplots(1, 3, figsize=(15, 15))
        ax[0].imshow(img)
        ax[0].set_title('Real image')
        ax[1].imshow(pred)
        ax[1].set_title('Predicted mask')
        ax[2].imshow(pred_thr)
        ax[2].set_title('Predicted mask with threshold')


def draw_hist(train_loss, train_dice, val_loss, val_dice):
    """Draws loss and dice metric curves.

    Parameters
    ----------
    train_loss : list[float]
    train_dice : list[float]
    val_loss : list[float]
    val_dice : list[float]

    Notes
    -----
    All parameters should have the same length

    """
    _, ax = plt.subplots(1, 2, figsize=(20, 8))
    ax[0].plot(train_loss, label="train loss")
    ax[0].plot(val_loss, label="val loss")
    ax[0].set_xlabel('Number of epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    ax[0].set_title('Loss while training and validating')

    ax[1].plot(train_dice, label="train dice")
    ax[1].plot(val_dice, label="val dice")
    ax[1].set_xlabel('Number of epoch')
    ax[1].set_ylabel('Dice')
    ax[1].legend()
    ax[1].set_title('Dice while training and validating')
    plt.show()


def eval_net(net, metric, threshold, path, device):
    """Returns predicted masks.

    Parameters
    ----------
    net : object of class(nn.Module)
        Net to evaluate.
    metric : function
        Metric to evaluate the model quality.
    threshold : float from 0 to 1
        Every predicted pixel with value > threshold corresponds to the cigarette.
    val_batch_gen : object of class torch.utils.data.DataLoader
        validation batch generator.
    device : cpu or gpu

    Returns
    -------
    masks : list[2d np.array]
        Predicted masks

    """
    net.eval()
    val_dice = 0
    masks = []
    trans_to_tens = transforms.ToTensor()
    trans_to_norm = transforms.Normalize([0.5315, 0.5226, 0.4595],
                                         [0.2005, 0.2006, 0.1882])
    s = nn.Sigmoid()
    img_names = sorted(os.listdir(f"{path}/images"))

    for idx in range(len(img_names)):

        img = Image.open(f"{path}/images/{img_names[idx]}").convert('RGB')

        sample = trans_to_norm(trans_to_tens(img).to(device)).unsqueeze(0)
        pred = s(net.forward(sample).data.cpu())
        pred = np.squeeze(pred.numpy(), axis=(0, 1))
        pred_thr = pred > threshold

        masks.append(pred_thr)

    return masks
