import torch
import torch.nn as nn
from typing import Iterable
from torch.utils.tensorboard import SummaryWriter
from utils.AverageMeter import AverageMeter
from utils.Accuracy import accuracy
from functools import reduce
from sklearn.metrics import confusion_matrix, classification_report

def train(model: torch.nn.Module, train_loader: Iterable,
          optimizer: torch.optim.Optimizer, epoch: int, summary: SummaryWriter):
    model.train()
    loss = nn.CrossEntropyLoss()
    train_loss = AverageMeter()
    for step, data in enumerate(train_loader):

        img, label = data
        label = label.cuda()
        img = img.float().cuda()
        pred = model(img)

        losses = loss(pred, label)
        train_loss.update(losses.item(), img.size()[0])
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    #print("losses : {} , epoch : {}".format(losses, epoch))

    summary.add_scalar('train/loss', train_loss.avg, epoch)

def val(model: torch.nn.Module, val_loader: Iterable, epoch: int, summary: SummaryWriter):
    model.eval()
    val_acc = AverageMeter()
    val_losses = AverageMeter()

    with torch.no_grad():
        loss = nn.CrossEntropyLoss()
        for step, data in enumerate(val_loader):
            img, label = data
            label = label.cuda()
            img = img.float()
            pred = model(img)
            losses = loss(pred, label)
            prec1 = accuracy(pred.data, label)[0]
            val_losses.update(losses.item(), img.size()[0])
            val_acc.update(prec1.item(), img.size()[0])
    print("val acc : {} epoch : {}".format(val_acc.avg, epoch))
    summary.add_scalar('val/loss', val_losses.avg, epoch)
    summary.add_scalar('val/acc', val_acc.avg, epoch)

def test(model: torch.nn.Module, val_loader: Iterable, epoch: int, summary: SummaryWriter):
    model.eval()
    val_acc = AverageMeter()
    val_losses = AverageMeter()
    temp = []
    pred_list = []

    with torch.no_grad():

        for step, data in enumerate(val_loader):
            image, label = data
            image = image.float().cuda()
            label = label.cuda()

            pred = model(image)

            prec1, preds = accuracy(pred.data, label)

            temp.append(label.tolist())
            pred_list.append(preds.tolist())

            val_acc.update(prec1.item(), image.size()[0])
    # confusion_matrix

    y_true = reduce(lambda x, y: x + y, temp)

    y_pred = []

    for i in pred_list:
        temp = i[0]
        for k in temp:
            y_pred.append(k)

    confusion_matrixs = confusion_matrix(y_true, y_pred)
    print(confusion_matrixs)
    print(classification_report(y_true, y_pred))