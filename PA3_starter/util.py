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
    

# [[[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[2, 2, 2], [2, 2, 2], [2, 2, 2]], [[3, 3, 3], [3, 3, 3], [3, 3, 3]]]