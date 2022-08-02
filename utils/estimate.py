import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from utils import AverageMeter,accuracy
from bloodnet50 import bloodnet50

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

valid_transform = transforms.Compose([
    transforms.CenterCrop(96),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1046, 0.0929, 0.0939], std=[0.1141, 0.1141, 0.1162]),   #这里重新计算了
])

valid_dataset = datasets.ImageFolder(
    root=opt.valid_folder,
    transform=valid_transform)
print(valid_dataset.class_to_idx)
ll=valid_dataset.targets
print(len(ll))
targets_to_one_hot = torch.nn.functional.one_hot(torch.tensor(ll))
print(targets_to_one_hot)
val_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

model =bloodnet50()
model.load_state_dict(torch.load('./bloodnet50_new.pth'))

model.eval()
tar = []
with torch.no_grad():
    for idx, (images, labels) in enumerate(val_loader):
        result = model(images)
        result = torch.nn.functional.softmax(result,dim=1)
        result = result.squeeze()
        tar.append(result)

