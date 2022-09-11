import sys
import time
from sklearn.metrics import r2_score
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from utils.utils import AverageMeter

from utils.lplot import LossHistory
from model.bloodnet import bloodnet50
import numpy as np
import argparse
parser = argparse.ArgumentParser('argument for regression training')
parser.add_argument('--print_freq', type=int, default=10,
                    help='print frequency')
parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')
# dataset
parser.add_argument('--data_folder', type=str, default='../data/train1/', help='path to custom dataset')
parser.add_argument('--valid_folder', type=str, default='../data/test/', help='path to valid dataset')
parser.add_argument('--weights', type=str, default='../weight/seresnet50-60a8950a85b2b.pkl', help='model weight')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--num_workers', type=int, default=6, help='num of workers to use')
parser.add_argument('--learning_rate', type=float, default=3e-4,
                    help='learning rate')
opt = parser.parse_args()

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
loss_history = LossHistory("./blood_loss_reg")
acc_history = LossHistory("./blood_r2_reg")
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
        valid_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)

#模型
model = bloodnet50(num_classes=1)
pretext_model = torch.load(opt.weights)
model2_dict = model.state_dict()
state_dict = {k: v for k, v in pretext_model.items() if k in model2_dict.keys() and np.shape(model2_dict[k]) == np.shape(v)}
model2_dict.update(state_dict)
model.load_state_dict(model2_dict)

model = model.to(device)
criterion = torch.nn.SmoothL1Loss(reduction='mean').to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

ll_ind = torch.tensor([14.0,1,21,28,7])

def train(epoch):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, label) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = ll_ind[label].cuda(non_blocking=True)
        bsz = labels.shape[0]
        output = model(images)
        output = torch.squeeze(output)
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        r2 = r2_score(labels.data.cpu().numpy(),output.data.cpu().numpy())
        top1.update(r2.item(), bsz)
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
                  'r2@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg

def validate():
    """one epoch training"""
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    with torch.no_grad():
        for idx, (images, label) in enumerate(val_loader):
            data_time.update(time.time() - end)

            images = images.cuda(non_blocking=True)
            labels = ll_ind[label].cuda(non_blocking=True)
            bsz = labels.shape[0]
            output = model(images)
            output = torch.squeeze(output)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            r2 = r2_score(labels.data.cpu().numpy(),output.data.cpu().numpy())
            top1.update(r2.item(), bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print info
            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'r2@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * r2@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg

def main():
    best_loss = 100.0


    # training routine
    for epoch in range(1, opt.epochs + 1):
        #adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss_train, r2 = train(epoch)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, r2:{:.2f}'.format(
            epoch, time2 - time1, r2))

        # eval for one epoch
        loss_val, val_r2 = validate()
        loss_history.append_loss(loss_train, loss_val)
        acc_history.append_loss(r2,val_r2)
        scheduler.step()

        if loss_val < best_loss:
            best_loss = loss_val
            torch.save(model.state_dict(), "bloodnet50_reg_my.pth")



if __name__ == '__main__':
    main()
