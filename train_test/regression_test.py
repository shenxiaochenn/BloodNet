import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from utils.utils import AverageMeter,accuracy

from model.bloodnet import bloodnet50
from utils.lplot import LossHistory
from sklearn.metrics import r2_score
import argparse
parser = argparse.ArgumentParser('argument for regression test')
parser.add_argument('--print_freq', type=int, default=10,
                    help='print frequency')
parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')
# dataset
parser.add_argument('--data_folder', type=str, default='../data/train1/', help='path to custom dataset')
parser.add_argument('--valid_folder', type=str, default='../data/test/', help='path to valid dataset')
parser.add_argument('--test_folder', type=str, default='../data/outside_test/', help='path to test dataset')
parser.add_argument('--weights', type=str, default='../weight/bloodnet50_reg.pth', help='model weight')
parser.add_argument('--batch_size', type=int, default=128, help='batch_size')
parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
parser.add_argument('--learning_rate', type=float, default=3e-4,
                    help='learning rate')
opt = parser.parse_args()

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



valid_transform = transforms.Compose([
    transforms.CenterCrop(96),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1046, 0.0929, 0.0939], std=[0.1141, 0.1141, 0.1162]),   #这里重新计算了
])

valid_dataset = datasets.ImageFolder(
    root=opt.valid_folder,
    transform=valid_transform)

test_dataset = datasets.ImageFolder(
    root=opt.test_folder,
    transform=valid_transform)

val_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True)


#模型
model = bloodnet50(1)
model.load_state_dict(torch.load(opt.weights))
model = model.to(device)
criterion = torch.nn.SmoothL1Loss(reduction='mean').to(device)
model.eval()

top1 = AverageMeter()
top2 = AverageMeter()
losses = AverageMeter()
ll_ind = torch.tensor([14.0,1,21,28,7])

with torch.no_grad():
    for idx, (images, label) in enumerate(test_loader):


        images = images.cuda(non_blocking=True)
        labels = ll_ind[label].cuda(non_blocking=True)
        bsz = labels.shape[0]
        output = model(images)
        output = torch.squeeze(output)
        loss = criterion(output, labels)

        losses.update(loss.item(), bsz)
        r2 = r2_score(labels.data.cpu().numpy(), output.data.cpu().numpy())
        top1.update(r2.item(), bsz)


print("R2:",top1.avg)
print("LOSS:",losses.avg)
