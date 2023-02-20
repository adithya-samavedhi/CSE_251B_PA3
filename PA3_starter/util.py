import numpy as np
import torch

def iou(pred, target, n_classes = 21):
    ious = []
    pred= torch.argmax(pred, dim =1)
    pred = pred.view(-1)
    target = target.view(-1)

    # Ignore IoU for undefined class ("9")
    for cls in range(n_classes-1):  # last class is ignored

        pred_inds = pred == cls
        target_inds = target == cls
        intersection = torch.sum(pred_inds & target_inds)
        union = torch.sum(pred_inds | target_inds)
        if float(union) == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection / union))


    return torch.nanmean(torch.Tensor(ious))

    # for c in range(n_classes):
    #     intersection = ((pred==c)*(target==c)).sum(dim = 1)
    #     union = ((pred==c)+(target==c)).sum(dim = 1)
    #     indices = torch.where(union!=0)
    #     intersection = intersection[indices]
    #     union = union[indices]
    #     #intersection = torch.Tensor([i for i,u in zip(intersection,union) if u!=0])
    #     #union = [u for u in union if u!=0]
    #     iou_scores.extend( torch.mean(intersection/union) )

    # return torch.mean(torch.Tensor(iou_scores))



def pixel_acc(pred, target):
    pred= torch.argmax(pred, dim =1)

    return (torch.sum(pred==target)/ pred.numel())*100
    
def plots(trainEpochLoss, trainEpochAccuracy, valEpochLoss, valEpochAccuracy, earlyStop):

    """
    Helper function for creating the plots
    earlyStop is the epoch at which early stop occurred and will correspond to the best model. e.g. earlyStop=-1 means the last epoch was the best one
    """

    fig1, ax1 = plt.subplots(figsize=((24, 12)))
    epochs = np.arange(1,len(trainEpochLoss)+1,1)
    ax1.plot(epochs, trainEpochLoss, 'r', label="Training Loss")
    ax1.plot(epochs, valEpochLoss, 'g', label="Validation Loss")
    plt.scatter(epochs[earlyStop],valEpochLoss[earlyStop],marker='x', c='g',s=400,label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10), fontsize=35 )
    plt.yticks(fontsize=35)
    ax1.set_title('Loss Plots', fontsize=35.0)
    ax1.set_xlabel('Epochs', fontsize=35.0)
    ax1.set_ylabel('Cross Entropy Loss', fontsize=35.0)
    ax1.legend(loc="upper right", fontsize=35.0)
    plt.savefig(constants.saveLocation+"loss.png")
    plt.show()

    fig2, ax2 = plt.subplots(figsize=((24, 12)))
    ax2.plot(epochs, trainEpochAccuracy, 'r', label="Training Accuracy")
    ax2.plot(epochs, valEpochAccuracy, 'g', label="Validation Accuracy")
    plt.scatter(epochs[earlyStop], valEpochAccuracy[earlyStop], marker='x', c='g', s=400, label='Early Stop Epoch')
    plt.xticks(ticks=np.arange(min(epochs),max(epochs)+1,10), fontsize=35)
    plt.yticks(fontsize=35)
    ax2.set_title('Accuracy Plots', fontsize=35.0)
    ax2.set_xlabel('Epochs', fontsize=35.0)
    ax2.set_ylabel('Accuracy', fontsize=35.0)
    ax2.legend(loc="lower right", fontsize=35.0)
    plt.savefig(constants.saveLocation+"accuracy.png")
    plt.show()
# [[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[2, 2, 2], [2, 2, 2], [2, 2, 2]], [[3, 3, 3], [3, 3, 3], [3, 3, 3]]]