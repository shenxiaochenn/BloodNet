import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

import torchvision

from d2l import torch as d2l
from bloodnet import *
def get_net():
    num_classes = 5
    net = blood_net(num_classes, 3)
    return net

a=get_net()

datadir="./your_data/" # can ask the author


transform_train = torchvision.transforms.Compose([

    torchvision.transforms.CenterCrop(96),

    torchvision.transforms.ToTensor(),

    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

transform_test = torchvision.transforms.Compose([

    torchvision.transforms.CenterCrop(96),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],
                                     [0.2023, 0.1994, 0.2010])])

train_ds= torchvision.datasets.ImageFolder(
    os.path.join(datadir, 'train_blood'),
    transform=transform_train)

test_ds= torchvision.datasets.ImageFolder(
    os.path.join(datadir, 'test_blood'),
    transform=transform_test)

batch_size=64
batch_size2=128

train_iter = torch.utils.data.DataLoader(train_ds, batch_size, shuffle=True,
                                         drop_last=True)

test_iter = torch.utils.data.DataLoader(test_ds, batch_size, shuffle=False,
                                        drop_last=False)


def evaluate_lossandacc_gpu(net, data_iter, device=None):
    loss = nn.CrossEntropyLoss()
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device

    metric = d2l.Accumulator(3)
    for X, y in data_iter:
        if isinstance(X, list):

            X = [x.to(device) for x in X]
        else:
            X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_hat = net(X)
        l = loss(y_hat, y)
        metric.add(l * y.numel(), d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def evaluate_accuracy_gpu(net, data_iter, device=None):
    if isinstance(net, torch.nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device

    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        if isinstance(X, list):

            X = [x.to(device) for x in X]
        else:
            X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def train_main(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'valid acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):

        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        torch.save(net.state_dict(), './network_%s_params.pth' % (epoch))
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'valid acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


def train_multi_gpu(net, num_gpus, train_iter, test_iter, num_epochs, lr,device):
    """multi_GPU。"""
    devices = [d2l.try_gpu(i) for i in range(num_gpus)]

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    # print('training on', device)
    # net.to(device)
    net = nn.DataParallel(net, device_ids=devices)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'valid acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):

        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device[0]), y.to(device[0])
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        # torch.save(net.state_dict(), './network_%s_params.pth'%(epoch))
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'valid acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')

if __name__ == '__main__':
    lr, num_epochs = 0.05, 1
    train_main(net=a, train_iter=train_iter, test_iter=test_iter, num_epochs=num_epochs, lr=lr,device='cuda')