import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pickle

class CustomModel(nn.Module):
    def __init__(self, n_class):
        super(CustomModel, self).__init__()

        # Use ResNet-101 as backbone
        self.n_class = n_class
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd6 = nn.BatchNorm2d(1024)
        self.conv7 = nn.Conv2d(1024, 2048, kernel_size=3, stride=2, padding=1, dilation=1)
        self.bnd7 = nn.BatchNorm2d(2048)

        self.relu = nn.ReLU(inplace=True)

        #self.resnet = models.resnet34(pretrained=True)

        # Remove the last two layers of the backbone
        #del self.resnet.layer4
        #del self.resnet.avgpool

        # Define the Atrous Spatial Pyramid Pooling (ASPP) module
        self.aspp = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=6, dilation=6),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=12, dilation=12),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=18, dilation=18),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        # Define the decoder module
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(256, 48, kernel_size=1),
        #     nn.BatchNorm2d(48),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(48, 48, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(48),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(48, n_class, kernel_size=1)
        # )
        

        # Initialize the weights of the decoder module
        # for m in self.decoder.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.normal_(m.weight, std=0.001)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)

        self.deconv1 = nn.ConvTranspose2d(1024, 1024, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=2, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(128,64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.deconv6 = nn.ConvTranspose2d(64,64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn6 = nn.BatchNorm2d(64)
        self.deconv7 = nn.ConvTranspose2d(64,32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn7 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    def forward(self, x):
        # Apply the encoder module
        x = self.bnd1(self.relu(self.conv1(x)))
        x = self.bnd2(self.relu(self.conv2(x)))
        x = self.bnd3(self.relu(self.conv3(x)))
        x = self.bnd4(self.relu(self.conv4(x)))
        x = self.bnd5(self.relu(self.conv5(x)))
        x = self.bnd6(self.relu(self.conv6(x)))
        x = self.bnd7(self.relu(self.conv7(x)))

        # Apply the ASPP module
        x = self.aspp(x)

        # Upsample the output of the ASPP module to the same size as the output of layer 2 in the backbone
        #x = F.interpolate(x, size=x.size()[2:4], mode='bilinear', align_corners=True)

        # Concatenate the upsampled output with the output of layer 2 in the backbone
        #x = torch.cat([x, self.resnet.layer2(x)], dim=1)

        # Apply the decoder module
        x = self.bn1(self.relu(self.deconv1(x)))
        x = self.bn2(self.relu(self.deconv2(x)))
        x = self.bn3(self.relu(self.deconv3(x)))
        x = self.bn4(self.relu(self.deconv4(x)))
        x = self.bn5(self.relu(self.deconv5(x)))
        x = self.bn6(self.relu(self.deconv6(x)))
        x = self.bn7(self.relu(self.deconv7(x)))

        score = self.classifier(x)

        # Upsample the output to the size of the input image
        #x = F.interpolate(x, size=x.size()[2:4], mode='bilinear', align_corners=True)

        return score

    def save(self, filepath="./model.pkl"):
        """
        Saves the Network model in the given filepath.
        Parameters
        ----------
        filepath: filepath of the model to be saved
        Returns
        -------
        None
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filepath="./model.pkl"):
        """
        Loads a pre-trained Network model from the given filepath.
        Parameters
        ----------
        filepath: filepath of the model to be loaded
        Returns
        -------
        model: Loaded Network model
        """
        with open(filepath, 'rb') as f:
            return pickle.load(f)
