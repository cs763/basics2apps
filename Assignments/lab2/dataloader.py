import numpy as np
import torch.utils.data as data
import os
from PIL import Image

class VOC(data.Dataset):
    def __init__(self, root, image_set):
        '''
            root: The root directory of the dataset
            image_set: 'train'|'val'|'test'
        '''
        self.root = root
        self.image_set = image_set
        self.image_size = 256
        voc_root = os.path.join(self.root, 'VOC2007')
        image_dir = os.path.join(voc_root, 'JPEGImages')
        mask_dir = os.path.join(voc_root, 'SegmentationClass')
        splits_dir = os.path.join(voc_root, 'ImageSets/Segmentation')

        split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        # split_f = os.path.join(splits_dir, 'train.txt')

        # Reading the image names from the train.txt file
        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".png") for x in file_names]
        '''
        if image_set == 'train':
            self.images = self.images[:180]
            self.masks = self.masks[:180]
        else:
            self.images = self.images[180:]
            self.masks = self.masks[180:]
        '''
        assert (len(self.images) == len(self.masks))


    def segments_to_masks(self, seg):
        '''
            PASCAL VOC gives a single image as ground truth in which every pixel is
            either 0(background), 1 -- 20 or 255(separation). In this function, we
            try to create 21 masks from them, each belonging to a classification category.
            seg: the input image
        '''
        seg_height = seg.shape[0]
        seg_width = seg.shape[1]
        masks = np.zeros((21, seg_height, seg_width))
        for i in range(seg_height):
            for j in range(seg_width):
                pixel_class = seg[i,j]

                # Since white lines are just for the sake of boundary, we will
                # predict them
                if pixel_class != 255:
                    masks[pixel_class, i, j] = 1
        return masks


    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        # Just trying to make all the images of the same size
        img = img.resize((self.image_size, self.image_size), Image.ANTIALIAS)
        img = np.array(img)
        img = img.transpose(2,0,1) / 255

        target = np.array(target)
        output_mask = self.segments_to_masks(target)

        # Similar to images, resize the corresponding segment masks
        output_mask = np.resize(output_mask, (21, self.image_size, self.image_size))
        return img, output_mask

    def __len__(self):
        '''
            A very important function. Helps the dataloader to know what range of
            indices should it randomly sample from
        '''
        return len(self.images)


if __name__ == "__main__":
    voc = VOC('./', 'train')
    train_loader = data.DataLoader(
            VOC('./', 'train'), batch_size = 2, shuffle = False)
    print(train_loader)

    for i, (img, mask) in enumerate(train_loader):
        print(img.size())
        print(mask.size())
        if i == 5:
            break

