import glob
import random
import os
import sys
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from utils.augmentations import horisontal_flip
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images

def sortByArea(df):
    col = (df.iloc[:, 5]-df.iloc[:, 3])*(df.iloc[:, 6]-df.iloc[:, 4])
    tmp = np.array(col.values.tolist())
    order = sorted(range(len(tmp)), key=lambda j: tmp[j], reverse=True)
    return df.iloc[order]

def collision(boxes, norm_sign):
    # boxes: idx, x_center, y_center, width, height
    # norm_sign: norm_x_center, norm_y_center, norm_width, norm_height
    norm_x_center, norm_y_center, norm_width, norm_height = norm_sign
    for box in boxes:
        if (norm_x_center-norm_width/2 <= box[1]+ box[3]/2 and norm_x_center+norm_width/2 >= box[1]- box[3]/2):
            if (norm_y_center-norm_height/2 <= box[2]+ box[4]/2 and norm_y_center+norm_height/2 >= box[2]- box[4]/2):
                return True
    return False

def preprocess(img, img_size=416):
    img = transforms.ToTensor()(img)
    img, _ = pad_to_square(img, 0)
    img = resize(img, img_size)
    return img

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(
        self,
        list_path,
        img_size=416,
        augment=True,
        multiscale=True,
        normalized_labels=True,
    ):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels")
            .replace(".png", ".txt")
            .replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor

        img = transforms.ToTensor()(Image.open(img_path).convert("RGB"))
        # img = transforms.ToTensor()(Image.open(img_path).convert("L"))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        # Gray scale
        if img.shape[0] == 1:
            img = torch.cat((img,) * 3, dim=0)

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        label_path = self.label_files[index % len(self.img_files)].rstrip()

        targets = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
            # Extract coordinates for unpadded + unscaled image
            x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
            y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
            x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
            y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
            # Adjust for added padding
            x1 += pad[0]
            y1 += pad[2]
            x2 += pad[1]
            y2 += pad[3]
            # Returns (x, y, w, h)
            boxes[:, 1] = ((x1 + x2) / 2) / padded_w
            boxes[:, 2] = ((y1 + y2) / 2) / padded_h
            boxes[:, 3] *= w_factor / padded_w
            boxes[:, 4] *= h_factor / padded_h

            targets = torch.zeros((len(boxes), 6))
            targets[:, 1:] = boxes
        # For images without labels
        else:
            targets = torch.empty(0, 6)

        # Apply augmentations
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)

class GTSDB_Dataset(Dataset):
    def __init__(
        self,
        list_path,
        img_size=416,
        augment=True,
        multiscale=True,
        normalized_labels=True,
        df_path = "data/custom/signs/GT-all.csv",
    ):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace("images", "labels")
            .replace(".png", ".txt")
            .replace(".jpg", ".txt")
            .replace(".ppm", ".txt")
            for path in self.img_files
        ]
        self.img_size = img_size
        self.max_objects = 100
        self.augment = augment
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.df = pd.read_csv(df_path)

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------

        img_path = self.img_files[index % len(self.img_files)].rstrip()

        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path).convert("RGB"))
        # img = transforms.ToTensor()(Image.open(img_path).convert("L"))

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        # Gray scale
        if img.shape[0] == 1:
            img = torch.cat((img,) * 3, dim=0)

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)

        # Sample from dataframe
        num_of_samples = random.randint(2, 10)
        df_samples = sortByArea(self.df.sample(num_of_samples))

        # Load label
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        boxes = None
        if os.path.exists(label_path):
            boxes = torch.from_numpy(np.loadtxt(label_path).reshape(-1, 5))
        else:
            boxes = torch.empty((0, 5))
        
        # --------------------------------------------
        #  Randomly paste sign images to street image
        # --------------------------------------------
        for i in range(len(df_samples)):
            sign_img = transforms.ToTensor()(Image.open(df_samples.iloc[i, 0]).convert("RGB"))
            x1, x2 = df_samples.iloc[i, 3], df_samples.iloc[i, 5]
            y1, y2 = df_samples.iloc[i, 4], df_samples.iloc[i, 6]
            sign_img = sign_img[:, y1:y2, x1:x2]
            _, sign_h, sign_w = sign_img.shape
            # Randomly generate coordinates for k times to avoild collision
            k = 3
            margin = 15
            while(k > 0):
                norm_sign_w = sign_w / w
                norm_sign_h = sign_h / h
                rand_x = random.randint(sign_w+margin, w-sign_w-margin)
                rand_y = random.randint(sign_h+margin, h-sign_h-margin)
                norm_x_center = rand_x/w + norm_sign_w/2
                norm_y_center = rand_y/h + norm_sign_h/2
                norm_sign = (norm_x_center, norm_y_center, norm_sign_w, norm_sign_h)
                # Check collision
                if collision(boxes, norm_sign):
                    k -= 1
                else:
                    new_box = torch.zeros((1, 5))
                    new_box[0, 0] = df_samples.iloc[i, -1]
                    new_box[0, 1] = norm_x_center
                    new_box[0, 2] = norm_y_center
                    new_box[0, 3] = norm_sign_w
                    new_box[0, 4] = norm_sign_h
                    boxes = torch.cat((boxes, new_box))
                    img[:, rand_y:rand_y+sign_img.shape[1], rand_x:rand_x+sign_img.shape[2]] = sign_img
                    break

        # ---------
        #  Padding
        # ---------
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # Modifiy label to based on square resolution
        # Extract coordinates for unpadded + unscaled image
        x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
        y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
        x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
        y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
        # Adjust for added padding
        x1 += pad[0]
        y1 += pad[2]
        x2 += pad[1]
        y2 += pad[3]
        # Returns (x, y, w, h)
        boxes[:, 1] = ((x1 + x2) / 2) / padded_w
        boxes[:, 2] = ((y1 + y2) / 2) / padded_h
        boxes[:, 3] *= w_factor / padded_w
        boxes[:, 4] *= h_factor / padded_h

        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = boxes

        # ---------------------
        #  Apply augmentations
        # ---------------------
        if self.augment:
            if np.random.random() < 0.5:
                img, targets = horisontal_flip(img, targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return len(self.img_files)