from basic_fcn import *
from resnet34 import *
from unet_architecture import *
import sys
import time
from torch.utils.data import DataLoader
import torch
import gc
import voc
from voc import *
import torchvision.transforms as standard_transforms
import torchvision
from torchvision import transforms, models
import torch.optim as optim
import torch.nn as nn
from torchsummary import summary
from util import *
import numpy as np
import torchvision.transforms.functional as TF
import argparse


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.normal_(m.bias.data)  # xavier not applicable for biases


# TODO Get class weights
def getClassWeights(train_dataset):
    dummy_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
    global_arr = torch.zeros(1, 21)
    for b in dummy_loader:
        curr_counts = torch.bincount(b[1][0].flatten()).reshape(1, -1)
        zeros_needed = 21 - curr_counts.shape[-1]

        # print(f"curr_outputs : {curr_counts}")

        x = torch.nn.functional.pad(curr_counts, (0, zeros_needed), "constant", 0)
        global_arr = global_arr + x

    h = torch.tensor([1, 2, 3])
    global_arr = global_arr.flatten()
    print(f"globl arr: {global_arr}")
    total_samples = sum(global_arr.tolist())
    print(f"total_samples: {total_samples}")
    global_arr = global_arr / total_samples
    global_arr = 1 - global_arr
    print("******** Class Weights *********", global_arr)
    # global_arr[15] = 0.2
    return torch.Tensor(global_arr)


def early_stopping(model, filepath, iter_num, early_stopping_rounds, best_loss, best_acc, best_iou, best_iter, loss,
                   acc, iou, patience):
    """
    Implements the early stopping functionality with a loss monitor. If the patience is exhausted it interupts the training process and
    returns the best model and its corresponding loss and accuracy score on validation data.
    Parameters
    ----------
    model: Trained model uptill iteration iter_num (epoch number)
    filepath: Path to save the model at if its the best model (in terms of validation loss) till the current iteration.
    iter_num: Current epoch number.
    early_stopping_rounds: User specified hyperparameters that indicates the patience period upper limit.
    best_loss: Best validation set loss observed till the current iteration.
    best_acc: Accuracy observed in the iteration with the best validation loss (best_loss).
    best_iter: Iteration number of best validation loss (best_loss)
    loss: Current iteration loss on validation data.
    acc: Current iteration accuracy on validation data.
    patience: Current patience level. If best_loss is not beaten then patience will be decremented by 1.
    Returns
    -------
    best_loss: Updated best loss after iteration iter_num.
    best_acc: Correspondingly updated best loss after iteration iter_num.
    best_iter: Best iteration till iter_num.
    patience: Updated patience value.
    """
    if loss >= best_loss:
        patience -= 1
    else:
        model.save(filepath)
        patience = early_stopping_rounds
        best_loss = loss
        best_acc = acc
        best_iou = iou
        best_iter = iter_num

    return best_loss, best_acc, best_iou, best_iter, patience


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
TF_transform = None


def transform(args):
    if args.transform == 'true':
        TF_transform = lambda x: [x, TF.hflip(x), TF.rotate(x.unsqueeze(0), angle=5, fill=0).squeeze(0),
                                  TF.rotate(x.unsqueeze(0), angle=5, fill=0).squeeze(0)]
        print("******* Applying Transformations ********", TF_transform)


original_train_dataset = voc.VOC('train', transform=input_transform, target_transform=target_transform)
original_val_dataset = voc.VOC('val', transform=input_transform, target_transform=target_transform)
original_test_dataset = voc.VOC('test', transform=input_transform, target_transform=target_transform)

transformed_train_dataset = voc.VOC('train', transform=input_transform, target_transform=target_transform,
                                    TF_transform=TF_transform)
transformed_val_dataset = voc.VOC('val', transform=input_transform, target_transform=target_transform,
                                  TF_transform=None)
transformed_test_dataset = voc.VOC('test', transform=input_transform, target_transform=target_transform,
                                   TF_transform=None)

train_dataset = torch.utils.data.ConcatDataset([transformed_train_dataset, original_train_dataset])
# val_dataset = torch.utils.data.ConcatDataset([transformed_val_dataset,original_val_dataset])
# test_dataset = torch.utils.data.ConcatDataset([transformed_test_dataset,original_test_dataset])
val_dataset = original_val_dataset
test_dataset = original_test_dataset

print(f"Training data: {len(train_dataset)}")
print(f"Validation data: {len(val_dataset)}")
print(f"Testing data: {len(test_dataset)}")

train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)
classWeights = getClassWeights(train_dataset)

epochs = 20
n_class = 21
global fcn_model

device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")  # TODO determine which device to use (cuda or cpu)
criterion = nn.CrossEntropyLoss().to(
    device)  # TODO Choose an appropriate loss function from https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html, weight=classWeights


# TODO
def train(args):
    scheduler = None

    optimizer = optim.Adam(fcn_model.parameters(), lr=0.0001)  # TODO choose an optimizer

    if args.scheduler == 'cosine':
        print("Using Cosine Learning Rate Scheduler")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=0, last_epoch=-1,
                                                               verbose=False)
    best_iou_score = 0.0
    best_pixel_acc = 0.0

    # Initializing early stopping parameters.
    if args.early_stop:
        patience = args.early_stop_epoch
        best_loss = 1e9
        best_iter = 0

    train_epoch_loss = []
    val_epoch_loss = []

    for epoch in range(epochs):
        ts = time.time()
        losses = []
        for iter, (inputs, labels) in enumerate(train_loader):

            optimizer.zero_grad()

            # both inputs and labels have to reside in the same device as the model's
            inputs = inputs.to(device)  # TODO transfer the input to the same device as the model's
            labels = labels.to(device)  # TODO transfer the labels to the same device as the model's

            outputs = fcn_model.forward(
                inputs)  # TODO  Compute outputs. we will not need to transfer the output, it will be automatically in the same device as the model's!
            loss = criterion(outputs, labels)  # TODO  calculate loss
            losses.append(loss.item())

            # TODO  backpropagate
            loss.backward()

            # TODO  update the weights
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}, Accuracy: {}, Mean IOU: {}".format(epoch, iter,
                                                                                     loss.item(),
                                                                                     pixel_acc(outputs, labels),
                                                                                     iou(outputs, labels)))

        if scheduler:
            scheduler.step()

        print("\nFinish epoch {}, time elapsed {}".format(epoch, time.time() - ts))

        current_loss, current_acc_score, current_miou_score = val(fcn_model, epoch)

        train_epoch_loss.append(np.mean(losses))
        val_epoch_loss.append(current_loss)

        # Check for Early Stopping
        if args.early_stop:
            best_loss, best_pixel_acc, best_iou_score, best_iter, patience = early_stopping(fcn_model, args.filepath,
                                                                                            epoch,
                                                                                            args.early_stop_epoch,
                                                                                            best_loss, best_pixel_acc,
                                                                                            best_iou_score,
                                                                                            best_iter, current_loss,
                                                                                            current_acc_score,
                                                                                            current_miou_score,
                                                                                            patience)
            print(f"Patience = {patience}")
            if patience == 0:
                print(
                    f"Training stopped early at epoch:{epoch}, best_loss = {best_loss}, best_pixel_acc = {best_pixel_acc}, best_iou_score = {best_iou_score}, best_iteration={best_iter}")
                break
        else:
            if current_miou_score > best_iou_score:
                best_iou_score = current_miou_score

            if current_acc_score > best_pixel_acc:
                best_pixel_acc = current_acc_score

    plots(train_epoch_loss, val_epoch_loss, best_iter, output_dir="./")


# TODO
def val(fcn_model, epoch):
    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

        for iter, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)  # TODO transfer the input to the same device as the model's
            labels = labels.to(device)  # TODO transfer the labels to the same device as the model's
            outputs = fcn_model.forward(inputs)
            loss = criterion(outputs, labels).item()
            # print(loss)
            acc = pixel_acc(outputs, labels)
            iou_score = iou(outputs, labels)

            losses.append(loss)
            accuracy.append(acc.cpu())
            mean_iou_scores.append(iou_score)

    print(f"Loss at epoch: {epoch} is {np.mean(losses)}")
    print(f"IoU at epoch: {epoch} is {np.mean(mean_iou_scores)}")
    print(f"Pixel acc at epoch: {epoch} is {np.mean(accuracy)}\n")

    fcn_model.train()  # TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!

    return np.mean(losses), np.mean(accuracy), np.mean(mean_iou_scores)


# TODO
def modelTest():
    global fcn_model
    fcn_model = fcn_model.load(args.filepath)
    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !

    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing

        for iter, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)  # TODO transfer the input to the same device as the model's
            labels = labels.to(device)  # TODO transfer the labels to the same device as the model's
            outputs = fcn_model.forward(inputs)
            loss = criterion(outputs, labels).item()
            acc = pixel_acc(outputs, labels)
            iou_score = iou(outputs, labels)

            losses.append(loss)
            accuracy.append(acc.cpu())
            mean_iou_scores.append(iou_score)

    print("\nTest metrics:")
    print(f"Loss: {np.mean(losses)}")
    print(f"IoU: {np.mean(mean_iou_scores)}")
    print(f"Pixel: {np.mean(accuracy)}")

    plotImages(fcn_model, test_loader)
    fcn_model.train()  # TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--scheduler', type=str, default='normal',
                        help='Specify the learning rate scheduler that you want to use. Out of [normal, cosine]')
    parser.add_argument('--model', type=str, default='transfer_learning',
                        help='Specify the model that you want to use. Out of [normal, unet, transfer_learning]')
    parser.add_argument('--filepath', type=str, default='model.pkl', help="Model path to save and load model from.")
    parser.add_argument('--early-stop', type=bool, default=True, help='Implement early stopping')
    parser.add_argument('--early-stop-epoch', type=int, default=3, help='Patience period of early stopping')
    parser.add_argument('--transform', type=str, default='false', help='Specify if you want to add transformations')
    args = parser.parse_args()

    if args.transform == 'true':
        transform(args)

    args.scheduler = "cosine"
    if args.model == 'normal':
        fcn_model = FCN(n_class=n_class)
        # fcn_model.apply(init_weights)
    elif args.model == 'unet':
        print("Using UNET architecture")
        fcn_model = UNet(3, n_class)
    elif args.model == 'transfer_learning':
        print("Using ResNet34 architecture for encoder")
        fcn_model = TransferLearningResNet34(n_class=n_class)

    fcn_model = fcn_model.to(device)

    val(fcn_model, 0)  # show the accuracy before training

    train(args)
    modelTest()

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()
