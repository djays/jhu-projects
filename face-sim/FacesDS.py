import random

import math
import skimage
import torch

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision import transforms

IMG_ROOT = 'lfw'


class XFRotation:
    def __call__(self, img):
        rot = random.randrange(-30, 30, 1)
        return img.rotate(rot)


class XFTranslation:
    def __call__(self, img):
        t_hr = random.randrange(-10, 10, 2)
        t_vr = random.randrange(-10, 10, 2)
        return img.transform(img.size, Image.AFFINE, (1, 0, t_hr, 0, 1, t_vr))


class XFScale:
    def __call__(self, img):
        x, y = img.size
        scale_factor = random.randrange(7, 13, 1) / 10.0
        rx, ry = int(round(scale_factor * x)), int(round(scale_factor * y))
        re_img = img.resize((rx, ry), Image.BILINEAR)
        if re_img.size[0] > 128:
            x1 = int(round((rx - 128) / 2.))
            y1 = int(round((ry - 128) / 2.))
            img = img.crop((x1, y1, x1 + 128, y1 + 128))
        elif re_img.size[0] < 128:
            # fill
            x, y = re_img.size
            img = Image.new('RGB', (128, 128))
            img.paste(re_img, (int((128 - x) / 2), int((128 - y) / 2)))
            img = img
        return img


class FacesDS(Dataset):
    XFORMS = (transforms.RandomHorizontalFlip, XFRotation, XFTranslation, XFScale)

    def __init__(self, datafile, augment=False):
        with open(datafile) as f:
            data = f.read().strip()
        self.data = [line.split() for line in data.split('\n')]
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def sk_xform(self, img):
        img = skimage.transform.resize(img, (128, 128), mode='constant')
        if self.augment and random.random() < 0.7:

            # Scale
            scale_factor = random.randrange(7, 13, 1) / 10.0

            # Flip
            if random.random() < 0.5:
                img = img[:, ::-1]

            # Rotate
            angle = math.radians(random.randrange(-30, 30))

            # Translate
            tr_x = random.randrange(-10, 10, 2)
            tr_y = random.randrange(-10, 10, 2)

            tform = skimage.transform.SimilarityTransform(scale=scale_factor, rotation=angle, translation=(tr_x, tr_y))
            img = skimage.transform.warp(img, inverse_map=tform)
        return torch.from_numpy(img.transpose((2, 0, 1))).type(torch.FloatTensor)

    def get_xform(self):
        xform = [transforms.ToPILImage(), transforms.Scale((128, 128))]
        if self.augment and random.random() < 0.7:
            xform += [xf() for xf in FacesDS.XFORMS]

        xform.append(transforms.ToTensor())
        return transforms.Compose(xform)

    def __getitem__(self, index):
        entry = self.data[index]

        im1 = io.imread('%s/%s' % (IMG_ROOT, entry[0]))
        im2 = io.imread('%s/%s' % (IMG_ROOT, entry[1]))
        #im1 = self.get_xform()(im1)
        #im2 = self.get_xform()(im2)
        im1 = self.sk_xform(im1)
        im2 = self.sk_xform(im2)
        return im1, im2, torch.FloatTensor([float(entry[2])])



#
# ds = FacesDS("train.txt", True)
# train_loader = DataLoader(ds, batch_size=8, shuffle=True)
#
# for i, e in enumerate(train_loader):
#     im = e[1][0]
#     im = transforms.ToPILImage()(im)
#     #print(im.size)
#     io.imsave("d:\\im.png", im)
