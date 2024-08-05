import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2
import torchvision.utils
from PIL.Image import Image
from torch.optim.lr_scheduler import LambdaLR

from models.facecolor_unet import Face_UNet
from models.textureGAN import Edge_UNet
from train_grad import computeGradient
from utils.data_loader import make_datapath_list, ImageDataset, ImageTransform
from models.FSRNet import Generator, Discriminator, GradientLossBlock
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torch.autograd import Variable
from collections import OrderedDict
from torchvision import models, transforms
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import time
import torch


torch.manual_seed(44)

# choose your device

def get_parser():
    parser = argparse.ArgumentParser(
        prog='ST-CGAN: Stacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal',
        usage='python3 main.py',
        description='This module demonstrates shadow detection and removal using ST-CGAN.',
        add_help=True)

    parser.add_argument('-e', '--epoch', type=int, default=10000, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('-l', '--load', type=str, default="170", help='the number of checkpoints')
    parser.add_argument('-hor', '--hold_out_ratio', type=float, default=0.8, help='training-validation ratio')
    parser.add_argument('-s', '--image_size', type=int, default=286)
    parser.add_argument('-cs', '--crop_size', type=int, default=256)
    parser.add_argument('-lr', '--lr', type=float, default=3e-5)
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--decay_epoch', type=int, default=50,help='epoch to start linearly decaying the learning rate to 0')

    return parser

def fix_model_state_dict(state_dict):
    '''
    remove 'module.' of dataparallel
    '''
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k
        if name.startswith('module.'):
            name = name[7:]
        new_state_dict[name] = v
    return new_state_dict

def set_requires_grad(nets, requires_grad=False):
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def unnormalize(x):
    x = x.transpose(1, 3)
    #mean, std
    x = x * torch.Tensor((0.5, )) + torch.Tensor((0.5, ))
    x = x.transpose(1, 3)
    return x

def evaluate(G1,Face,GRAD, dataset, device, filename):
    img, gt_shadow, gt,face = zip(*[dataset[i] for i in range(11)])
    img = torch.stack(img)
    gt_shadow = torch.stack(gt_shadow)
    gt = torch.stack(gt)
    face = torch.stack(face)

    with torch.no_grad():
        grad, featureMap = GRAD(img.to(device))
        grad = transforms.Grayscale()(grad)
        grad = grad.to(torch.device('cpu'))
        facemap, facefeatureMap = Face(img.to(device))
        facemap = facemap.to(torch.device('cpu'))
        # shadowfree=shadowfree+gt_shadow
        # gt=gt+gt_shadow

        detected_shadow = G1(img.to(device),facefeatureMap,featureMap)
        # detected_shadow = G1(img.to(device),facefeatureMap)
        detected_shadow = detected_shadow.to(torch.device('cpu'))

    grid_detect = make_grid(torch.cat((unnormalize(detected_shadow), unnormalize(gt)), dim=0))
    torchvision.utils.save_image(grid_detect, filename+'_detect12.09.jpg')
    # save_image(grid_detect, filename + '_grid11.30.jpg')
    torchvision.utils.save_image(unnormalize(grad), filename + '_GRAD12.09.png')
    torchvision.utils.save_image(unnormalize(facemap), filename + '_facemap12.09.png')
    # save_image(detected_shadow, filename + '_face11.30.jpg')

def check_dir():
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    if not os.path.exists('./result'):
        os.mkdir('./result')
    if not os.path.exists('./results'):
        os.mkdir('./results')

def train_model(G1,D1,Face,GRAD, dataloader, val_dataset, num_epochs, parser, save_model_name='model'):

    check_dir()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    G1.to(device)
    D1.to(device)
    GRAD.to(device)
    Face.to(device)

    """use GPU in parallel"""
    if device == 'cuda':
        G1 = torch.nn.DataParallel(G1)
        print("parallel mode")

    print("device:{}".format(device))

    lr = parser.lr

    beta1, beta2 = 0.5, 0.999

    optimizerG = torch.optim.Adam([{'params': G1.parameters()}],
                                  lr=lr,
                                  betas=(beta1,beta2))
    optimizerD = torch.optim.Adam([{'params': D1.parameters()}],
                                  lr=lr,
                                  betas=(beta1, beta2))
    # lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=LambdaLR(parser.n_epochs, parser.epoch,
    #                                                                                    parser.decay_epoch).step)
    # lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda=LambdaLR(parser.n_epochs, parser.epoch,
    #                                                                                    parser.decay_epoch).step)
    # criterionGAN = nn.MSELoss().to(device)
    criterionL1 = nn.L1Loss().to(device)
    criterionSoftplus = nn.Softplus().to(device)
    # pgrad_loss = 0.1
    torch.backends.cudnn.benchmark = True

    mini_batch_size = parser.batch_size
    num_train_imgs = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    lambda_dict = {'lambda1':5,  'lambda3':0.1}

    iteration = 1
    g_losses = []
    d_losses = []
    # grad_losses=[]
    for epoch in range(num_epochs+1):
        G1.train()
        D1.train()
        GRAD.eval()
        Face.eval()

        t_epoch_start = time.time()

        epoch_g_loss = 0.0
        epoch_d_loss = 0.0

        print('-----------')
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('(train)')

        for images, imgmatting, gt,face in tqdm(dataloader):

            # if size of minibatch is 1, an error would be occured.
            if images.size()[0] == 1:
                continue

            images = images.to(device)
            gt = gt.to(device)
            imgmatting = imgmatting.to(device)
            face = face.to(device)
            # gt=gt+imgmatting
            facemap, facefeaturemap = Face(images)
            grad, featuremap = GRAD(images)
            grad = transforms.Grayscale()(grad)
            # Train Discriminator
            set_requires_grad([D1], True)  # enable backprop$
            optimizerD.zero_grad()
            # recovergrad=torch.concat([images,gt_shadow],dim=1)
            # print(recovergrad.shape)

            # shadowfree=shadowfree+imgmatting

            finalfree = G1(images,facefeaturemap,featuremap)
            # finalfree = G1(images,facefeaturemap)
            # finalfree=finalfree+imgmatting
            # torchvision.utils.save_image(unnormalize(grad.to(torch.device('cpu'))),
            #                              '{:s}/val_{:d}'.format('results', epoch) + '_detect12.08freeimage.png')
            # torchvision.utils.save_image(unnormalize(facemap.to(torch.device('cpu'))),
            #                              '{:s}/val_{:d}'.format('results', epoch) + '_detect12.09freeimage.png')
            fake1 = torch.cat([images, finalfree], dim=1)
            real1 = torch.cat([images, gt], dim=1)

            out_D1_fake = D1(fake1.detach())
            out_D1_real = D1(real1.detach())
            batchsize, _, w, h = out_D1_fake.size()
            # .detach() is not required as real1 doesn't have grad
            # gradgt=gradblock(connel=3)(gt)
            # L_CGAN1
            label_D1_fake = Variable(Tensor(np.zeros(out_D1_fake.size())), requires_grad=True)
            label_D1_real = Variable(Tensor(np.ones(out_D1_fake.size())), requires_grad=True)

            # loss_D1_fake = criterionGAN(out_D1_fake, label_D1_fake)
            loss_D1_fake = torch.sum(criterionSoftplus(out_D1_fake)) / batchsize / w / h
            # loss_D1_real = criterionGAN(out_D1_real, label_D1_real)
            loss_D1_real = torch.sum(criterionSoftplus(-out_D1_real)) / batchsize / w / h

            D_L_CGAN1 = loss_D1_fake + loss_D1_real
            # total
            D_loss =  lambda_dict['lambda3']*D_L_CGAN1
            # D_loss =  D_L_CGAN1
            D_loss.backward()
            optimizerD.step()

            # Train Generator
            set_requires_grad([D1], False)
            optimizerG.zero_grad()
            # L_CGAN1
            # newimages = torch.concat([images, grad], dim=1)
            fake1 = torch.cat([images, finalfree], dim=1)
            out_D1_fake = D1(fake1.detach())
            G_L_CGAN1 =torch.sum(criterionSoftplus(-out_D1_fake)) / batchsize / w / h
            # G_L_CGAN1 =criterionGAN(out_D1_fake,label_D1_real)

            G_L_data1=criterionL1(gt,finalfree)

            # max_pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
            # shadow_mask_dilate = -max_pool(gt_shadow)

            # pgrad_loss = GradientLoss()(gradimage,gt)
            # # G_loss= lambda_dict['lambda1'] *G_L_data1  + lambda_dict['lambda3'] * G_L_CGAN1

            # gradgt = GradientLossBlock()(gt)
            # # 转换成灰度图
            # gradgt = transforms.Grayscale()(gradgt)

            G_loss= lambda_dict['lambda1']*G_L_data1  +  lambda_dict['lambda3']*G_L_CGAN1
            # print(G_loss)
            G_loss.backward()
            optimizerG.step()

            epoch_d_loss += D_loss.item()
            epoch_g_loss += G_loss.item()

            torchvision.utils.save_image(unnormalize(finalfree.to(torch.device('cpu'))),
                                         '{:s}/val_{:d}'.format('results', epoch) + '_detect11.15facegradimage.png')
            # print(epoch_g_loss)
        t_epoch_finish = time.time()
        print('-----------')
        print('epoch {} || Epoch_D_Loss:{:.4f} || Epoch_G_Loss:{:.4f}'.format(epoch, epoch_d_loss/batch_size, epoch_g_loss/batch_size))
        print('timer: {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))

        d_losses += [epoch_d_loss / batch_size]
        g_losses += [epoch_g_loss / batch_size]


        t_epoch_start = time.time()
        # plot_log({'G': g_losses, 'D': d_losses}, save_model_name)

        if (epoch % 10 == 0):
            torch.save(G1.state_dict(), 'checkpoints/' + save_model_name + '_G1_5.20NoATTENTIONface' + str(epoch) + '.pth')
            # torch.save(D1.state_dict(), 'checkpoints/' + save_model_name + '_D1_12.08face' + str(epoch+110) + '.pth')
            G1.eval()
            GRAD.eval()
            Face.eval()
            evaluate(G1,Face,GRAD, val_dataset, device, '{:s}/val_{:d}'.format('results', epoch))

    return G1,D1

def main(parser):
    G1 = Generator(in_channel=3, out_channel=3)
    D1 = Discriminator(input_channels=6)
    GRAD=Edge_UNet(feature=True)
    Face=Face_UNet(feature=False)
    # Shadowremove= Edge_UNet1()

    '''load'''
    if parser.load is not None:
        print('load checkpoint ' + parser.load)
        # G1_weights = torch.load('./checkpoints/ST-CGAN_G1_12.08face'+parser.load+'.pth')
        GRAD_weights = torch.load('./checkpoints/TRAIN_GRID_1.6Edge_'+parser.load+'.pth')
        Face_weights = torch.load('./checkpoints/TRAIN_GRID_1.6Face_260.pth')
        # Shadowremove_weights = torch.load('./checkpoints/TRAIN_GRID_11.30free_'+parser.load+'.pth')
        # G1.load_state_dict(fix_model_state_dict(G1_weights))
        GRAD.load_state_dict(fix_model_state_dict(GRAD_weights))
        Face.load_state_dict(fix_model_state_dict(Face_weights))
        # Shadowremove.load_state_dict(fix_model_state_dict(Shadowremove_weights))

    train_img_list, val_img_list = make_datapath_list(phase='train', rate=parser.hold_out_ratio)

    mean = (0.5,)
    std = (0.5,)
    size = parser.image_size
    crop_size = parser.crop_size
    batch_size = parser.batch_size
    num_epochs = parser.epoch

    train_dataset = ImageDataset(img_list=train_img_list,
                                img_transform=ImageTransform(size=size, crop_size=crop_size, mean=mean, std=std),
                                phase='train')
    val_dataset = ImageDataset(img_list=val_img_list,
                                img_transform=ImageTransform(size=size, crop_size=crop_size, mean=mean, std=std),
                                phase='val')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #num_workers=4

    G1_update, G2_update, D1_update, D2_update = train_model(G1,D1,Face,GRAD, dataloader=train_dataloader,
                                                            val_dataset=val_dataset, num_epochs=num_epochs,
                                                            parser=parser, save_model_name='ST-CGAN')

if __name__ == "__main__":
    parser = get_parser().parse_args()
    main(parser)