import os
import torch.utils.data as data
from . import ISTD_transforms
from PIL import Image
import random
from torchvision import transforms
import matplotlib.pyplot as plt


def make_datapath_list(phase="train", rate=0.8):
    """
    make filepath list for train, validation and test images
    """
    random.seed(44)

    rootpath = './dataset/' + phase + '/'
    files_name = os.listdir(rootpath + phase + '_E')

    if phase=='train':
        random.shuffle(files_name)
    elif phase=='test':
        files_name.sort()

    path_E = []
    path_F = []
    path_G = []
    path_H = []

    for name in files_name:
        path_E.append(rootpath + phase + '_E/'+name)
        path_F.append(rootpath + phase + '_F/'+name)
        path_G.append(rootpath + phase + '_G/'+name)
        path_H.append(rootpath + phase + '_H/'+name)

    num = len(path_E)

    if phase=='train':
        path_E, path_E_val = path_E[:int(num*rate)], path_E[int(num*rate):]
        path_F, path_F_val = path_F[:int(num*rate)], path_F[int(num*rate):]
        path_G, path_G_val = path_G[:int(num*rate)], path_G[int(num*rate):]
        path_H, path_H_val = path_H[:int(num*rate)], path_H[int(num*rate):]
        path_list = {'path_E': path_E, 'path_F': path_F, 'path_G': path_G,'path_H': path_H}
        path_list_val = {'path_E': path_E_val, 'path_F': path_F_val, 'path_G': path_G_val, 'path_H': path_H_val}
        return path_list, path_list_val

    elif phase=='test':
        path_list = {'path_E': path_E, 'path_F': path_F, 'path_G': path_G, 'path_H': path_H}
        return path_list

class ImageTransformOwn():
    """
    preprocessing images for own images
    """
    def __init__(self, size=256, mean=(0.5, ), std=(0.5, )):
        self.data_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(mean, std)])

    def __call__(self, img):
        return self.data_transform(img)


class ImageTransform():
    """
    preprocessing images
    """
    def __init__(self, size=286, crop_size=256, mean=(0.5, ), std=(0.5, )):
        self.data_transform = {'train': ISTD_transforms.Compose([
                                                            ISTD_transforms.ToTensor(),
                                                            ISTD_transforms.Normalize(mean, std)],
        ),

                                'val': ISTD_transforms.Compose([
                                                           ISTD_transforms.ToTensor(),
                                                           ISTD_transforms.Normalize(mean, std)]),

                                'test': ISTD_transforms.Compose([
                                                            ISTD_transforms.ToTensor(),
                                                            ISTD_transforms.Normalize(mean, std)])}

    def __call__(self, phase, img):
        return self.data_transform[phase](img)


class ImageDataset(data.Dataset):
    """
    Dataset class. Inherit Dataset class from PyTrorch.
    """
    def __init__(self, img_list, img_transform, phase):
        self.img_list = img_list
        self.img_transform = img_transform
        self.phase = phase

    def __len__(self):
        return len(self.img_list['path_E'])

    def __getitem__(self, index):
        '''
        get tensor type preprocessed Image
        '''
        img = Image.open(self.img_list['path_E'][index]).convert('RGB')
        imgmatting = Image.open(self.img_list['path_F'][index]).convert('RGB')
        gt = Image.open(self.img_list['path_G'][index]).convert('RGB')
        face = Image.open(self.img_list['path_H'][index]).convert('RGB')

        img, imgmatting, gt,face = self.img_transform(self.phase, [img, imgmatting, gt,face])

        return img, imgmatting, gt,face

if __name__ == '__main__':
    img = Image.open('../dataset/train/train_A/test.png').convert('RGB')
    gt_shadow = Image.open('../dataset/train/train_B/test.png')
    gt = Image.open('../dataset/train/train_C/test.png').convert('RGB')

    print(img.size)
    print(gt_shadow.size)
    print(gt.size)

    f = plt.figure()
    f.add_subplot(1, 3, 1)
    plt.imshow(img)
    f.add_subplot(1, 3, 2)
    plt.imshow(gt_shadow, cmap='gray')
    f.add_subplot(1, 3, 3)
    plt.imshow(gt)

    img_transforms = ImageTransform(size=286, crop_size=256, mean=(0.5, ), std=(0.5, ))
    img, gt_shadow, gt = img_transforms([img, gt_shadow, gt])

    print(img.shape)
    print(gt_shadow.shape)
    print(gt.shape)


    f.add_subplot(2, 3, 4)
    plt.imshow(transforms.ToPILImage()(img).convert('RGB'))
    f.add_subplot(2, 3, 5)
    plt.imshow(transforms.ToPILImage()(gt_shadow).convert('L'), cmap='gray')
    f.add_subplot(2, 3, 6)
    plt.imshow(transforms.ToPILImage()(gt).convert('RGB'))
    f.tight_layout()
    plt.show()