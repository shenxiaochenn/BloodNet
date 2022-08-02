import sys


import time

import torch.nn as nn

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from utils import AverageMeter,accuracy
# from model import bloodnet
from model_other import  senet34, resnet34
from lplot import LossHistory
# from model_senet import se_resnet50
# from bloodnet50 import bloodnet50
# from bloodnet50_huan import bloodnet50
# from bloodnet50_ca import bloodnet50
# from  bloodnet50_sa import bloodnet50
from torchvision.models.resnet import resnet50
from swin_transformer import  swin_tiny_patch4_window7_224
import argparse
parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--print_freq', type=int, default=10,
                    help='print frequency')
parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')
# dataset
parser.add_argument('--data_folder', type=str, default='./train1/', help='path to custom dataset')
parser.add_argument('--valid_folder', type=str, default='./outside_test/', help='path to valid dataset')
parser.add_argument('--batch_size', type=int, default=1, help='batch_size')
parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')

parser.add_argument('--learning_rate', type=float, default=3e-4,
                    help='learning rate')
opt = parser.parse_args()

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# valid_transform = transforms.Compose([
#     transforms.CenterCrop(96),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.1046, 0.0929, 0.0939], std=[0.1141, 0.1141, 0.1162]),   #这里重新计算了
# ])
#[0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010]
valid_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1046, 0.0929, 0.0939], std=[0.1141, 0.1141, 0.1162]),   #这里重新计算了
])

valid_dataset = datasets.ImageFolder(
    root=opt.valid_folder,
    transform=valid_transform)

val_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

#模型
# model = bloodnet(5, 3)
# model.load_state_dict(torch.load('./bloodnet.pth'))
# model = senet34(5,3)
# model = resnet34(5,3)
# model.load_state_dict(torch.load('./senet.pth'))
# model.load_state_dict(torch.load('./resnet.pth'))
# model = se_resnet50()
# class senet(nn.Module):
#     def __init__(self, ):
#         super(senet, self).__init__()
#         self.encoder = torch.nn.Sequential(*list(model.children())[:-1])
#         self.head = nn.Linear(2048, 5)
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = torch.flatten(x, 1)
#         x = self.head(x)
#         return x
#     def freeze_backbone(self, ):
#         for param in self.encoder.parameters():
#             param.requires_grad = False
#
# model = senet()
# model.load_state_dict(torch.load('./bloodnet_pre.pth'))
# model =bloodnet50()
# model.load_state_dict(torch.load('./bloodnet50_ca.pth'))
# model_ = resnet50()
# class resnet(nn.Module):
#     def __init__(self, ):
#         super(resnet, self).__init__()
#         self.encoder = torch.nn.Sequential(*list(model_.children())[:-1])
#         self.head = nn.Linear(2048, 5)
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = torch.flatten(x, 1)
#         x = self.head(x)
#         return x
#     def freeze_backbone(self, ):
#         for param in self.encoder.parameters():
#             param.requires_grad = False
#
# model = resnet()
# model.load_state_dict(torch.load('./res50_finetune_ff.pth'))
model = swin_tiny_patch4_window7_224(num_classes=5)
model.load_state_dict(torch.load('swin_new_pre.pth'))
model = model.to(device)

model.eval()

top1 = AverageMeter()
top2 = AverageMeter()
with torch.no_grad():
    for idx, (images, labels) in enumerate(val_loader):


        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        output = model(images)
        acc1, acc2 = accuracy(output, labels, topk=(1, 2))
        top1.update(acc1[0].item(), bsz)
        top2.update(acc2[0].item(), bsz)

print(top1.avg)
print(top2.avg)