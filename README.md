# CSE_251B_PA3

## Code structure summary

1. basic_fcn.py has the code for the baseline model.
2. Dataset Download: run download.py to download the data. The dataset will be downloaded in 'data' directory 
4. train.py: This file contains the training loop with early stopping and we define loss criterion, optimizer, cosine annealing learning rate scheduler.
5. voc.py : This file creates the datset using Pytorch's dataset class. Input transformations can be applied by providing the argument --transform true.

6. util.py: We calculate the iOU and the mean pixel accuracy. It also contains the code to generate plots.
7. custom.py: contains the custom architecture that builds upon our baseline model.
8. resnet34.py: contains the transfer learning architecture using resnet 34 as the base.
9. unet_architecture.py: contains the u-net architecture.

# Instructions to run
python CSE_251B_PA3/PA3_starter/download.py # One-time dataset download

python CSE_251B_PA3/PA3_starter/train.py [args]

The arguments and their corresponding values are as follows:
1. scheduler: normal (default) , cosine 
2. model: normal(default), transfer_learning, unet
3. filepath:
4. early-stop: True(default), False
5. early_stop-epoch: 3 (default) [This defines the patience]
