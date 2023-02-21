import os
from PIL import Image
from torch.utils import data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import numpy as np

num_classes = 21
ignore_label = 255
root = './data'

'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle # 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''


#Feel free to convert this palette to a map
palette = [0, 0, 0, 128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128, 128, 0, 128, 0, 128, 128,
           128, 128, 128, 64, 0, 0, 192, 0, 0, 64, 128, 0, 192, 128, 0, 64, 0, 128, 192, 0, 128,
           64, 128, 128, 192, 128, 128, 0, 64, 0, 128, 64, 0, 0, 192, 0, 128, 192, 0, 0, 64, 128]  #3 values- R,G,B for every class. First 3 values for class 0, next 3 for
#class 1 and so on......

device =   torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def plotImages(fcn_model,test_loader, output_dir='./'):
    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    count = 0
 
    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing
 
        for iter, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)  # TODO transfer the input to the same device as the model's
            labels = labels.to(device)  # TODO transfer the labels to the same device as the model's
            print("input shape", inputs.shape)
            outputs = fcn_model.forward(inputs)
            print("output shape before", outputs.shape)
            outputs = torch.argmax(outputs, dim = 1)
            print("output unique",np.unique(outputs.cpu().numpy(), return_counts=True))
            print("output shape after", outputs.shape)
 
            inputImage = (inputs[0].permute(1, 2, 0).cpu().numpy())
            imgplot = plt.imshow(inputImage)
            fig_name = f"{output_dir}Input{count}.jpg"
            plt.savefig(fig_name)
            plt.show()

            output = outputs[0]
            print("output values", output)
            new_predictions = np.zeros((output.shape[0], output.shape[1], 3))
            rows, cols = output.shape[0], output.shape[1]
            # Rearranging the image here
            for row in range(rows):
                for col in range(cols):
                    idx = int(output[row][col])
                    new_predictions[row][col][:] = np.asarray(palette[idx*3:idx*3+3])
 
            new_predictions = new_predictions/255
            imgplot = plt.imshow(new_predictions)
            fig_name = f"{output_dir}SegmentedOutput{count}.jpg"
            plt.savefig(fig_name)
            plt.show()

            ##### Target print #####

            output = labels[0]
            print("output shapeee", output.shape)
            new_predictions = np.zeros((output.shape[0], output.shape[1], 3))
            rows, cols = output.shape[0], output.shape[1]
            # Rearranging the image here
            for row in range(rows):
                for col in range(cols):
                    idx = int(output[row][col])
                    new_predictions[row][col][:] = np.asarray(palette[idx*3:idx*3+3])
 
            new_predictions = new_predictions/255
            imgplot = plt.imshow(new_predictions)
            fig_name = f"{output_dir}SegmentedTarget{count}.jpg"
            plt.savefig(fig_name)
            plt.show()
            count +=1
            if count >10:
              break


def make_dataset(mode):
    assert mode in ['train', 'val', 'test']
    items = []
    if mode == 'train':
        img_path = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        mask_path = os.path.join(root, 'VOCdevkit', 'VOC2007', 'SegmentationClass')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Segmentation', 'train.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)
    elif mode == 'val':
        img_path = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        mask_path = os.path.join(root, 'VOCdevkit', 'VOC2007', 'SegmentationClass')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Segmentation', 'val.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)
    else:
        img_path = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
        mask_path = os.path.join(root, 'VOCdevkit', 'VOC2007', 'SegmentationClass')
        data_list = [l.strip('\n') for l in open(os.path.join(
            root, 'VOCdevkit', 'VOC2007', 'ImageSets', 'Segmentation', 'test.txt')).readlines()]
        for it in data_list:
            item = (os.path.join(img_path, it + '.jpg'), os.path.join(mask_path, it + '.png'))
            items.append(item)
    return items


class VOC(data.Dataset):
    def __init__(self, mode, transform=None, target_transform=None, TF_transform=None):
        self.imgs = make_dataset(mode)
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.TF_transform = TF_transform
        self.width = 224
        self.height = 224

    def __getitem__(self, index):

        img_path, mask_path = self.imgs[index]
        img = Image.open(img_path).convert('RGB').resize((self.width, self.height))
        mask = Image.open(mask_path).resize((self.width, self.height))

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        if self.TF_transform is not None:
            img = self.TF_transform(img)
            mask = self.TF_transform(mask)
            

        mask[mask==ignore_label]=0

        return img, mask

    def __len__(self):
        return len(self.imgs)