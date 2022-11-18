import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as tf
from torch.utils.data import DataLoader

from resnet import ResNet18

from tqdm.auto import tqdm
from torchsummary import summary
from matplotlib import pyplot as plt
import matplotlib.ticker as m_tick
from torch.optim.lr_scheduler import StepLR

device = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int):
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_weights(m: nn.Module):
    class_name = m.__class__.__name__
    if class_name.find('Conv2d') != -1 or class_name.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif class_name.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def get_loader(train_path, val_path, batch_size, workers):
    imagenet_norm_mean = (0.485, 0.456, 0.406)
    imagenet_norm_std = (0.229, 0.224, 0.225)

    # Can do Data augmentation here
    t_transform = tf.Compose([
        tf.Resize((64, 64)),
        # reverse
        tf.RandomHorizontalFlip(),
        tf.ToTensor(),
        # 标准化是把图片3个通道中的数据整理到规范区间 x = (x - mean(x))/stdd(x)
        # [0.485, 0.456, 0.406]这一组平均值是从imagenet训练集中抽样算出来的,对应3个channel
        tf.Normalize(imagenet_norm_mean, imagenet_norm_std)
    ])
    # Validation don't need any operation
    v_transform = tf.Compose([
        tf.Resize((64, 64)),
        tf.ToTensor(),
        tf.Normalize(imagenet_norm_mean, imagenet_norm_std)
    ])
    # ImageFloder instead of 'class getData()'
    train_dataset = tv.datasets.ImageFolder(root=train_path, transform=t_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)

    val_dataset = tv.datasets.ImageFolder(root=val_path, transform=v_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)

    return train_loader, val_loader


def draw_acc_loss(train_acc, val_acc, train_loss, val_loss):
    x1 = np.arange(args.epochs)

    fig = plt.figure(1)
    # Set Y as type of %
    ax1 = fig.add_subplot()

    fmt = '%.2f%%'
    y_ticks = m_tick.FormatStrFormatter(fmt)
    ax1.yaxis.set_major_formatter(y_ticks)
    # plt.figure(figsize=(9, 6), dpi=300)

    ax1.plot(x1, 100 * train_acc.reshape(-1), label='train_acc')
    ax1.plot(x1, 100 * val_acc.reshape(-1), '-', label='val_acc')

    ax1.set_ylabel('acc')
    ax1.set_xlabel('iter')
    ax1.set_ylim([0, 100])  # 设置y轴取值范围

    # This is the important function, twin image.
    ax2 = ax1.twinx()

    ax2.set_ylim([0, 5])  # 设置y轴取值范围
    ax2.set_ylabel('loss')

    ax2.plot(x1, train_loss.reshape(-1), '--', label='train_loss')
    ax2.plot(x1, val_loss.reshape(-1), '--', label='val_loss')

    # The loc of description
    ax1.legend(loc=(1 / 32, 16 / 19))
    ax2.legend(loc=(1 / 32, 12 / 19))

    plt.savefig('/output/iters.png')
    # plt.show()


def main(args: argparse.Namespace):
    print('---------Train on: ' + device + '----------')

    # Set random seed
    if args.seed is not None:
        set_seed(args.seed)

    # Loading data and transform. ImageNet -- mean & std
    train_loader, val_loader = get_loader(args.train_path, args.val_path, args.batch_size, args.workers)

    # Create model
    model = ResNet18().to(device)
    # We don't use init_weight here -- some bugs.
    # model.apply(init_weights)

    # Visualize model
    summary(model, input_size=(3, 64, 64))

    # Define Optimizer and Loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Define list to record acc & loss for plt
    train_loss = np.array([])
    train_acc = np.array([])
    val_loss = np.array([])
    val_acc = np.array([])

    # For epoch in range(args.epochs):
    for epoch in range(args.epochs):
        # train
        train_batch_loss, train_batch_acc = train(epoch, train_loader, model, optimizer, criterion, args)
        train_loss = np.append(train_loss, train_batch_loss)
        train_acc = np.append(train_acc, train_batch_acc)
        # validate
        val_batch_loss, val_batch_acc = validate(epoch, val_loader, model, criterion, args)
        val_loss = np.append(val_loss, val_batch_loss)
        val_acc = np.append(val_acc, val_batch_acc)
        # Save model
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'seed': args.seed
            }, './model/model_{}.pth'.format(epoch + 1))

    # Draw loss & acc
    draw_acc_loss(train_acc, val_acc, train_loss, val_loss)


def train(epoch: int, train_loader: DataLoader, model, optimizer, criterion, args: argparse.Namespace):
    model.train()
    train_loss_lis = np.array([])
    train_acc_lis = np.array([])
    for batch in tqdm(train_loader):
        imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)
        # labels = torch.nn.functional.one_hot(labels).long().to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        # Compute gradient and do GD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate the batch acc
        acc = (logits.argmax(dim=-1) == labels).float().mean()

        # Record the batch loss and accuracy.
        train_loss_lis = np.append(train_loss_lis, loss.item())
        train_acc_lis = np.append(train_acc_lis, acc.cpu())

    train_loss = sum(train_loss_lis) / len(train_loss_lis)
    train_acc = sum(train_acc_lis) / len(train_acc_lis)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{args.epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")
    return train_loss, train_acc


def validate(epoch: int, val_loader: DataLoader, model, criterion, args: argparse.Namespace):
    model.eval()
    val_loss_lis = np.array([])
    val_acc_lis = np.array([])
    for batch in tqdm(val_loader):
        imgs, labels = batch

        with torch.no_grad():
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)
            # Calculate the batch acc
            acc = (logits.argmax(dim=-1) == labels).float().mean()

            # Record the batch loss and accuracy.
            val_loss_lis = np.append(val_loss_lis, loss.item())
            val_acc_lis = np.append(val_acc_lis, acc.cpu())
    val_loss = sum(val_loss_lis) / len(val_loss_lis)
    val_acc = sum(val_acc_lis) / len(val_acc_lis)

    # Print the information.
    print(f"[ Validation | {epoch + 1:03d}/{args.epochs:03d} ]  acc = {val_acc:.5f}")
    return val_loss, val_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Source for ImageNet Classification')
    parser.add_argument('-sd', '--seed', default=17, type=int, help='seed for initializing training. ')

    # dataset parameters
    parser.add_argument('-tp', '--train_path', default='/data/bitahub/Tiny-ImageNet/train', help='the path of training data.')
    parser.add_argument('-vp', '--val_path', default='/data/bitahub/Tiny-ImageNet/val_reorg', help='the path of validation data.')
    parser.add_argument('-wn', '--workers', type=int, default=2, help='number of data loading workers (default: 2)')

    # train parameters
    parser.add_argument('-bs', '--batch_size', type=int, default=256, help='the size of batch.')
    parser.add_argument('-ep', '--epochs', type=int, default=20, help='the num of epochs.')

    # model parameters
    parser.add_argument('-lr', '--lr', type=float, default=0.001, help='initial learning rate', dest='lr')
    parser.add_argument('-mm', '--momentum', type=float, default=0.9, help='initial momentum')
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-5, help='initial momentum')

    args = parser.parse_args()

    main(args)