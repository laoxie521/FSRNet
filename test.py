import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import cv2
import math
from models.facecolor_unet import Face_UNet
from models.textureGAN import Edge_UNet
# from models.shadowgrad_unet import Edge_UNet1
from utils.data_loader import make_datapath_list, ImageDataset, ImageTransform, ImageTransformOwn
from models.FSRNet import Generator
from torchvision.utils import make_grid
from torchvision.utils import save_image
from torchvision import transforms
from collections import OrderedDict
from PIL import Image
import numpy as np
import argparse
import time
import torch


# import models.RRDBNet_arch as arch
torch.manual_seed(44)
# choose your device



def get_parser():
    parser = argparse.ArgumentParser(
        prog='ST-CGAN: Stacked Conditional Generative Adversarial Networks for Jointly Learning Shadow Detection and Shadow Removal',
        usage='python3 main.py',
        description='This module demonstrates shadow detection and removal using ST-CGAN.',
        add_help=True)

    parser.add_argument('-l', '--load', type=str, default=None, help='the number of checkpoints')
    parser.add_argument('-i', '--image_path', type=str, default=None, help='file path of image you want to test')
    parser.add_argument('-o', '--out_path', type=str, default='./test_result', help='saving path')
    parser.add_argument('-s', '--image_size', type=int, default=286)
    parser.add_argument('-cs', '--crop_size', type=int, default=256)
    parser.add_argument('-rs', '--resized_size', type=int, default=256)

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


def check_dir():
    if not os.path.exists('./test_result'):
        os.mkdir('./test_result')
    if not os.path.exists('./test_result/detected_shadow'):
        os.mkdir('./test_result/detected_shadow')
    if not os.path.exists('./test_result/shadow_removal_image'):
        os.mkdir('./test_result/shadow_removal_image')
    if not os.path.exists('./test_result/grid'):
        os.mkdir('./test_result/grid')
def tensor2img(tensor, out_type=np.float32, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.float32:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)

def unnormalize(x):
    x = x.transpose(1, 3)
    # mean, std
    x = x * torch.Tensor((0.5,)) + torch.Tensor((0.5,))
    x = x.transpose(1, 3)
    return x

def save_img(img, img_path, mode='RGB'):
    # skimage.io.imsave(img_path,img)
    cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    # torchvision.utils.save_image(img, img_path)
def test(G1,Face,GRAD, test_dataset):
    '''
    this module test dataset from ISTD dataset
    '''
    check_dir()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    G1.to(device)
    GRAD.to(device)
    Face.to(device)
    # model.to(device)

    """use GPU in parallel"""
    if device == 'cuda':
        G1 = torch.nn.DataParallel(G1)
        GRAD = torch.nn.DataParallel(GRAD)
        Face = torch.nn.DataParallel(Face)
        # G2 = torch.nn.DataParallel(G2)
        print("parallel mode")

    print("device:{}".format(device))

    G1.eval()
    GRAD.eval()
    Face.eval()
    # model.eval()
    # G2.eval()

    for n, (img, gt_shadow, gt,face) in enumerate([test_dataset[i] for i in range(test_dataset.__len__())]):
        print(test_dataset.img_list['path_E'][n].split('/')[4][:-4])

        img = torch.unsqueeze(img, dim=0)
        gt_shadow = torch.unsqueeze(gt_shadow, dim=0).to(device)
        gt = torch.unsqueeze(gt, dim=0)
        face = torch.unsqueeze(face, dim=0)
        # concat = torch.cat([img, gt_shadow], dim=1)
        with torch.no_grad():
            grad, featuremap = GRAD(img.to(device))
            grad = transforms.Grayscale()(grad)
            facemap, facefeaturemap = Face(img.to(device))
            # finalfree = G1(img.to(device),featuremap,facefeaturemap)
            finalfree = G1(img.to(device),facefeaturemap,featuremap)
            finalfree=finalfree.to(torch.device('cpu'))
           
        save_img(tensor2img(gt_shadow)+tensor2img(finalfree), './test_result/grid/' + test_dataset.img_list['path_E'][n].split('/')[4])
      
def test_own_image(G1, G2, path, out_path, size, img_transform):
    img = Image.open(path).convert('RGB')
    width, height = img.width, img.height
    img = img.resize((size, size), Image.LANCZOS)
    img = img_transform(img)
    img = torch.unsqueeze(img, dim=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    G1.to(device)
    G2.to(device)

    """use GPU in parallel"""
    if device == 'cuda':
        G1 = torch.nn.DataParallel(G1)
        G2 = torch.nn.DataParallel(G2)
        print("parallel mode")

    print("device:{}".format(device))

    G1.eval()
    G2.eval()

    with torch.no_grad():
        detected_shadow = G1(img.to(device))
        detected_shadow = detected_shadow.to(torch.device('cpu'))
        concat = torch.cat([img, detected_shadow], dim=1)
        shadow_removal_image = G2(concat.to(device))
        shadow_removal_image = shadow_removal_image.to(torch.device('cpu'))

        grid = make_grid(torch.cat([unnormalize(img),
                                    unnormalize(torch.cat([detected_shadow, detected_shadow, detected_shadow], dim=1)),
                                    unnormalize(shadow_removal_image)],
                                   dim=0))

        save_image(grid, out_path + '/grid_' + path.split('/')[-1])

        detected_shadow = transforms.ToPILImage(mode='L')(unnormalize(detected_shadow)[0, :, :, :])
        detected_shadow = detected_shadow.resize((width, height), Image.LANCZOS)
        detected_shadow.save(out_path + '/detected_shadow_' + path.split('/')[-1])

        shadow_removal_image = transforms.ToPILImage(mode='RGB')(unnormalize(shadow_removal_image)[0, :, :, :])
        shadow_removal_image = shadow_removal_image.resize((width, height), Image.LANCZOS)
        shadow_removal_image.save(out_path + '/shadow_removal_image_' + path.split('/')[-1])


def main(parser):
    G1 = Generator(in_channel=3, out_channel=3)
    # G2 = Generator(input_channels=4, output_channels=3)
    GRAD = Edge_UNet(feature=True)
    Face = Face_UNet(feature=False)
    # model = arch.RRDBNet(3, 3, 64, 23, gc=32)
    '''load'''
    if parser.load is not None:
        print('load checkpoint ' + parser.load)
        G1_weights = torch.load('./checkpoints/ST-CGAN_G1_5.20NoATTENTIONface' + parser.load + '.pth')
        GRAD_weights = torch.load('./checkpoints/TRAIN_GRID_1.6Edge_170.pth')
        Face_weights = torch.load('./checkpoints/TRAIN_GRID_1.6Face_260.pth')

        # model_weights = torch.load('./checkpoints/RRDB_ESRGAN_x4.pth')
        G1.load_state_dict(fix_model_state_dict(G1_weights))
        GRAD.load_state_dict(fix_model_state_dict(GRAD_weights))
        Face.load_state_dict(fix_model_state_dict(Face_weights))

    mean = (0.5,)
    std = (0.5,)

    size = parser.image_size
    crop_size = parser.crop_size
    resized_size = parser.resized_size

    # test own image
    if parser.image_path is not None:
        print('test ' + parser.image_path)
        test_own_image(G1, parser.image_path, parser.out_path, resized_size,
                       img_transform=ImageTransformOwn(size=size, mean=mean, std=std))

    # test images from the ISTD dataset
    else:
        print('test ISTD dataset')
        test_img_list = make_datapath_list(phase='test')
        test_dataset = ImageDataset(img_list=test_img_list,
                                    img_transform=ImageTransform(size=size, crop_size=crop_size, mean=mean, std=std),
                                    phase='test')
        test(G1,Face,GRAD, test_dataset)


if __name__ == "__main__":
    parser = get_parser().parse_args()
    main(parser)