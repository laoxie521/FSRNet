import argparse
import os
import time

import numpy as np
import torch
from collections import OrderedDict

import torchvision
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
import torch.nn.functional as F

from models.FSRNet import color_loss
# from models.grad_unet import Edge_UNet
from models.grad_unet import Face_UNet
from models.textureGAN import Edge_UNet
from utils.data_loader1 import ImageDataset, ImageTransform, make_datapath_list
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(44)
# choose your device
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def get_parser():
    parser = argparse.ArgumentParser(
        prog='ST-CGAN: Stacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal',
        usage='python3 main.py',
        description='This module demonstrates shadow detection and removal using ST-CGAN.',
        add_help=True)

    parser.add_argument('-e', '--epoch', type=int, default=10000, help='Number of epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('-l', '--load', type=str, default=None, help='the number of checkpoints')
    parser.add_argument('-hor', '--hold_out_ratio', type=float, default=0.8, help='training-validation ratio')
    parser.add_argument('-s', '--image_size', type=int, default=286)
    parser.add_argument('-cs', '--crop_size', type=int, default=256)
    # parser.add_argument('-lr', '--lr', type=float, default=1e-3)
    parser.add_argument('-lr', '--lr', type=float, default=3e-5)
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

def computeGradient(pred):
    _, cin, _, _ = pred.shape
    # _, cout, _, _ = target.shape
    assert cin == 3
    kx = torch.Tensor([[1, 0, -1], [2, 0, -2],
                       [1, 0, -1]]).view(1, 1, 3, 3).to(pred)
    ky = torch.Tensor([[1, 2, 1], [0, 0, 0],
                       [-1, -2, -1]]).view(1, 1, 3, 3).to(pred)
    kx = kx.repeat((cin, 1, 1, 1))
    ky = ky.repeat((cin, 1, 1, 1))

    pred_grad_x = F.conv2d(pred, kx, padding=1, groups=3)
    pred_grad_y = F.conv2d(pred, ky, padding=1, groups=3)
    # target_grad_x = F.conv2d(target, kx, padding=1, groups=3)
    # target_grad_y = F.conv2d(target, ky, padding=1, groups=3)

    # loss = (
    #     nn.L1Loss(reduction=self.reduction)
    #     (pred_grad_x, target_grad_x) +
    #     nn.L1Loss(reduction=self.reduction)
    #     (pred_grad_y, target_grad_y))
    return pred_grad_x,pred_grad_y
    # torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2)

    # image_grad = np.zeros((1, image.shape[0], image.shape[1]))
    # gradients = np.gradient(image)
    # x_grad = gradients[0]
    # y_grad = gradients[1]
    # image_grad[0, :, :] = np.sqrt(np.power(x_grad, 2) + np.power(y_grad, 2))
    #
    # return image_grad
def PoissonGradientLoss(target):
    f = torch.Tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).view(1, 1, 3, 3).to(target)
    f = f.repeat((3, 1, 1, 1))
    grad_t = F.conv2d(target, f, padding=1, groups=3)
    return grad_t

def bgr2gray(bgr):
    bgr = bgr.data.cpu().numpy()
    b, g, r = bgr[0, :, :], bgr[1, :, :], bgr[2, :, :]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def normalization(x):
    maxi = np.max(x)
    mini = np.min(x)
    x = (x - mini) / (maxi - mini)

    return x
def unnormalize(x):
    x = x.transpose(1, 3)
    #mean, std
    x = x * torch.Tensor((0.5, )) + torch.Tensor((0.5, ))
    x = x.transpose(1, 3)
    return x
def check_dir():
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    if not os.path.exists('./result'):
        os.mkdir('./result')
def evaluate(Face,Edge, dataset, device, filename):
    img, gt_shadow, gt,face = zip(*[dataset[i] for i in range(8)])
    img = torch.stack(img)
    gt = torch.stack(gt)
    face = torch.stack(face)
    with torch.no_grad():
        batch_size = gt.shape[0]
        gt = gt.type(torch.FloatTensor)
        gt = Variable(gt).cuda()
        gt_grad = PoissonGradientLoss(gt)
        gt_grad = transforms.Grayscale()(gt_grad)

        B_edge_gt = gt_grad.to(torch.device('cpu'))
        # # B_edge_gt = Variable(B_edge_gt).cuda()
        grad,featuremap = Edge(img.to(device))
        grad = transforms.Grayscale()(grad)
        grad = grad.to(torch.device('cpu'))
        gt = gt.to(torch.device('cpu'))
        facemap,facefeature = Face(img.to(device))
       
        facemap = facemap.to(torch.device('cpu'))
        # face = face.to(torch.device('cpu'))
        # gt = gt.to(torch.device('cpu'))

    grid_detect = make_grid(torch.cat((unnormalize(grad),unnormalize(B_edge_gt)), dim=0))
    face_detect = make_grid(torch.cat((unnormalize(facemap), unnormalize(face)), dim=0))
    # grid_detect = make_grid(torch.cat((unnormalize(grad), dim=0))
    save_image(grid_detect, filename+'_grid_detect12.08.jpg')
    save_image(face_detect, filename+'_edge_detect12.08.jpg')

def train_gridmodel(Face,Edge, dataloader, val_dataset, num_epochs, parser, save_model_name='model'):
    check_dir()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    lr = parser.lr
    opt_edge = optim.Adam(Edge.parameters(), lr=lr)
    opt_free = optim.Adam(Face.parameters(), lr=lr)
    """use GPU in parallel"""
    if device == 'cuda':
        Face = torch.nn.DataParallel(Face)
        Edge = torch.nn.DataParallel(Edge)
        print("parallel mode")

    print("device:{}".format(device))
    g1_losses = []
    face_losses = []

    for epoch in range(num_epochs + 1):
        Face.train()
        Edge.train()
        loss_gpu_num = 1
        l1 = nn.L1Loss().cuda(loss_gpu_num)
        mse = nn.MSELoss().cuda(loss_gpu_num)
        t_epoch_start = time.time()
        acc_edge_L1_loss = 0
        acc_face_loss=0
        for images, gt_shadow, gt,face in tqdm(dataloader):
            if images.size()[0] == 1:
                continue

            images = images.to(device)
            gt = gt.to(device)
            gt_shadow = gt_shadow.to(device)
            face = face.to(device)
            batch_size = gt.shape[0]
            # B_edge_gt = np.zeros((batch_size, 1, gt.shape[2], gt.shape[3]))
            # for i in range(batch_size):

            # Compute gradient ground truth
            # gt = torch.from_numpy(gt).cuda()
            gt = gt.type(torch.FloatTensor)
            gt = Variable(gt).cuda()
            #evaluate EdgeGradient
            gt_grad = PoissonGradientLoss(gt)
            # print(gt_grad.shape)
            gt_grad = transforms.Grayscale()(gt_grad)

            # Extract Edge
            images = Variable(images).cuda()
            B_edge, featureMap = Edge(images)
            B_edge = transforms.Grayscale()(B_edge)
            # print(B_edge.shape)
            # Compute losses
            opt_edge.zero_grad()
            edge_L1_loss = mse(B_edge, gt_grad) * batch_size
            edge_L2_loss=l1(B_edge,gt_grad)*batch_size
            edge_loss = edge_L1_loss+edge_L2_loss
            # # Update edge extractor
            edge_loss.backward()
            opt_edge.step()
            # Extract Face
            facemap, featureMap = Face(images)       
            opt_free.zero_grad()

            face_L1_loss=color_loss(facemap,face)*batch_size

            face_L1_loss.backward()
            opt_free.step()

            acc_face_loss += face_L1_loss.data
            acc_edge_L1_loss += edge_loss.data

        t_epoch_finish = time.time()
        print('-----------')
        print('epoch {} || Epoch_edge_Loss:{:.4f} '.format(epoch, acc_face_loss / batch_size))
        print('timer: {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        g1_losses += [acc_edge_L1_loss / batch_size]
        face_losses += [acc_face_loss / batch_size]
        # g1_losses += [acc_edge_L2_loss / batch_size]

        if(epoch%10 == 0):
            torch.save(Edge.state_dict(), 'checkpoints/' + save_model_name + '_1.6Edge_' + str(epoch) + '.pth')
            torch.save(Face.state_dict(), 'checkpoints/'+save_model_name+'_1.6Face_'+str(epoch)+'.pth')
            Face.eval()
            Edge.eval()
            evaluate(Face,Edge, val_dataset, device, '{:s}/val_{:d}'.format('result', epoch))

def main(parser):
    Face = Face_UNet(feature=False).cuda()
    Edge = Edge_UNet(feature=True)
    train_img_list, val_img_list = make_datapath_list(phase='train', rate=parser.hold_out_ratio)

    mean = (0.5,)
    std = (0.5,)
    size = parser.image_size
    crop_size = parser.crop_size
    batch_size = parser.batch_size
    num_epochs = parser.epoch
    if parser.load is not None:
        print('load checkpoint ' + parser.load)
        # Edge_weights = torch.load('./checkpoints/TRAIN_GRID_1.6Edge_'+parser.load+'.pth')
        # Edge.load_state_dict(fix_model_state_dict(Edge_weights))
        # Face_weights = torch.load('./checkpoints/TRAIN_GRID_1.6Face_370.pth')
        # Face.load_state_dict(fix_model_state_dict(Face_weights))

    train_dataset = ImageDataset(img_list=train_img_list,
                                img_transform=ImageTransform(size=size, crop_size=crop_size, mean=mean, std=std),
                                phase='train')
    val_dataset = ImageDataset(img_list=val_img_list,
                                img_transform=ImageTransform(size=size, crop_size=crop_size, mean=mean, std=std),
                                phase='val')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #num_workers=4

    G1_update, G2_update, D1_update, D2_update = train_gridmodel(Face,Edge,dataloader=train_dataloader,
                                                            val_dataset=val_dataset, num_epochs=num_epochs,
                                                            parser=parser, save_model_name='TRAIN_GRID')

if __name__ == "__main__":
    parser = get_parser().parse_args()
    main(parser)