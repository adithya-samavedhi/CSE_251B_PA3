from basic_fcn import *
from unet_architecture import *
import sys
import time
from torch.utils.data import DataLoader
import torch
import gc
import voc
import torchvision.transforms as standard_transforms
import torchvision 
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from util import *
import numpy as np
import argparse

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases



#TODO Get class weights
def getClassWeights(train_dataset):
    dummy_loader = DataLoader(dataset=train_dataset, batch_size= 1, shuffle=True)
    global_arr = torch.zeros(1,21)
    for b in dummy_loader:

        curr_counts = torch.bincount(b[1][0].flatten()).reshape(1,-1)
        zeros_needed = 21- curr_counts.shape[-1]

        x = torch.nn.functional.pad(curr_counts,(0,zeros_needed),"constant",0)
        global_arr = global_arr+x
    h = torch.tensor([1,2,3])

    return 1/global_arr


    raise NotImplementedError


mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
input_transform = standard_transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(degrees=45),
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

target_transform = MaskToTensor()

train_dataset =voc.VOC('train', transform=input_transform, target_transform=target_transform)
val_dataset = voc.VOC('val', transform=input_transform, target_transform=target_transform)
test_dataset = voc.VOC('test', transform=input_transform, target_transform=target_transform)


print(f"Training data: {len(train_dataset)}")
print(f"Validation data: {len(val_dataset)}")
print(f"Testing data: {len(test_dataset)}")


train_loader = DataLoader(dataset=train_dataset, batch_size= 16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size= 16, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size= 16, shuffle=False)
classWeights = getClassWeights(train_dataset)


epochs =20

n_class = 21

fcn_model = FCN(n_class=n_class)
fcn_model.apply(init_weights)

device =   torch.device("cuda" if torch.cuda.is_available() else "cpu")# TODO determine which device to use (cuda or cpu)

optimizer = optim.Adam(fcn_model.parameters(), lr=0.001) # TODO choose an optimizer
criterion = nn.CrossEntropyLoss(weight=classWeights) # TODO Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html

fcn_model =  fcn_model.to(device) # TODO transfer the model to the device


# TODO
def train(args):
    scheduler = None
    if args.scheduler == 'cosine':
        print("Using Cosine Learning Rate Scheduler")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=0, last_epoch=-1, verbose=False)
    best_iou_score = 0.0

    if args.model == 'unet':
        print("Using UNET architecture")
        fcn_model = UNet(3,n_class)

    for epoch in range(epochs):
        ts = time.time()
        for iter, (inputs, labels) in enumerate(train_loader):

            # print(f"inputs:  {inputs.size()}")
            # print(f"labels:  {labels.size()}")
            # print(torch.max(labels[0]))
            # print(torch.min(labels[0]))
            # TODO  reset optimizer gradients
            optimizer.zero_grad()

            # both inputs and labels have to reside in the same device as the model's
            inputs =  inputs.to(device)# TODO transfer the input to the same device as the model's
            labels =   labels.to(device) # TODO transfer the labels to the same device as the model's

            outputs =  fcn_model.forward(inputs) # TODO  Compute outputs. we will not need to transfer the output, it will be automatically in the same device as the model's!

            loss =  criterion(outputs, labels) #TODO  calculate loss

            # TODO  backpropagate
            loss.backward()

            # TODO  update the weights
            optimizer.step()
        if scheduler:
            scheduler.step()
            


            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}, Accuracy: {}, Mean IOU: {}".format(epoch, iter, 
                loss.item(), pixel_acc(outputs, labels), iou(outputs, labels)))

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))

        current_miou_score = val(epoch)

        if current_miou_score > best_iou_score:
            best_iou_score = current_miou_score
            # save the best model
    
 #TODO
def val(epoch):
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing

        for iter, (inputs, labels) in enumerate(val_loader):
            inputs =  inputs.to(device)# TODO transfer the input to the same device as the model's
            labels =   labels.to(device) # TODO transfer the labels to the same device as the model's
            outputs =  fcn_model.forward(inputs)
            loss =  criterion(outputs, labels).item()
            acc = pixel_acc(outputs, labels)
            iou_score = iou(outputs, labels)

            losses.append(loss)
            accuracy.append(acc)
            mean_iou_scores.append(iou_score)

    print(f"Loss at epoch: {epoch} is {np.mean(losses)}")
    print(f"IoU at epoch: {epoch} is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc at epoch: {epoch} is {np.mean(accuracy)}")

    fcn_model.train() #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(mean_iou_scores)

 #TODO
def modelTest():
    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

        for iter, (input, label) in enumerate(test_loader):
            inputs =  inputs.to(device)# TODO transfer the input to the same device as the model's
            labels =   labels.to(device) # TODO transfer the labels to the same device as the model's
            outputs =  fcn_model.forward(inputs)
            loss =  criterion(outputs, labels).item()
            acc = pixel_acc(outputs, labels)
            iou_score = iou(outputs, labels)

            losses.append(loss)
            accuracy.append(acc)
            mean_iou_scores.append(iou_score)

    print(f"Loss: {np.mean(losses)}")
    print(f"IoU: {np.mean(mean_iou_scores)}")
    print(f"Pixel: {np.mean(accuracy)}")

    fcn_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--scheduler', type=str, default='cosine', help='Specify the learning rate scheduler that you want to use')
    parser.add_argument('--model', type=str, default='basic_fcn', help = 'Specify the model that you want to use')

    args = parser.parse_args()

    val(0)  # show the accuracy before training
    train(args)
    modelTest()

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()