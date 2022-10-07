import sys
sys.path.append("../")

import time

import torch.nn as nn
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from utils.utils import AverageMeter,accuracy
from model.bloodnet import bloodnet50
from utils.lplot import LossHistory
import argparse
parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--print_freq', type=int, default=10,
                    help='print frequency')
parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
# dataset
parser.add_argument('--data_folder', type=str, default='../data/train1/', help='path to custom dataset')
parser.add_argument('--valid_folder', type=str, default='../data/test/', help='path to valid dataset')
parser.add_argument('--weights', type=str, default='../weight/bloodnet50_new.pth', help='model weight')
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
parser.add_argument('--learning_rate', type=float, default=3e-4,
                    help='learning rate')
opt = parser.parse_args()

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_history = LossHistory("./blood_loss")
acc_history = LossHistory("./blood_acc")
#数据
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(96),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1046, 0.0929, 0.0939], std=[0.1141, 0.1141, 0.1162]),
])

valid_transform = transforms.Compose([
    transforms.CenterCrop(96),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1046, 0.0929, 0.0939], std=[0.1141, 0.1141, 0.1162]),
])

train_dataset= datasets.ImageFolder(
    root=opt.data_folder,
    transform=train_transform)

valid_dataset = datasets.ImageFolder(
    root=opt.valid_folder,
    transform=valid_transform)

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

#模型
model = bloodnet50()

for m in model.modules():
    if isinstance(m, torch.nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

pretext = torch.load(opt.weights)


model.load_state_dict(pretext)
model = model.to(device)
criterion = torch.nn.CrossEntropyLoss()
criterion = criterion.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)

def train(epoch):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        output = model(images)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc2 = accuracy(output, labels, topk=(1,2))
        top1.update(acc1[0].item(), bsz)
        # adam
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg

def validate(epoch):
    """one epoch training"""
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            data_time.update(time.time() - end)

            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            bsz = labels.shape[0]
            output = model(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc2 = accuracy(output, labels, topk=(1,2))
            top1.update(acc1[0].item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg

def main():
    best_loss = 3.0


    # training routine
    for epoch in range(1, opt.epochs + 1):

        # train for one epoch
        time1 = time.time()
        loss_train, acc = train(epoch)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, acc))

        # eval for one epoch
        loss_val, val_acc = validate(epoch)
        loss_history.append_loss(loss_train, loss_val)
        acc_history.append_loss(acc,val_acc)
        scheduler.step()

        if loss_val < best_loss:
            best_loss = loss_val
            torch.save(model.state_dict(), "my_new.pth")

    print('best accuracy: {:.2f}'.format(best_loss))

if __name__ == '__main__':
    main()

