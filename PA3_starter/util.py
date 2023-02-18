import numpy as np
import torch

def iou(pred, target, n_classes = 21):
    # pred = pred.view(-1)
    pred= torch.argmax(pred, dim =1)
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)
    # taget = pred.view(-1)
    iou_scores = []
    for c in range(n_classes):
        intersection = ((pred==c)*(target==c)).sum(dim = 1)
        union = ((pred==c)+(target==c)).sum(dim = 1)
        indices = torch.where(union!=0)
        intersection = intersection[indices]
        union = union[indices]
        #intersection = torch.Tensor([i for i,u in zip(intersection,union) if u!=0])
        #union = [u for u in union if u!=0]
        iou_scores.extend(intersection/union)

    return torch.mean(torch.Tensor(iou_scores))



def pixel_acc(pred, target):
    pred= torch.argmax(pred, dim =1)

    return (torch.sum(pred==target)/ pred.numel())*100
    