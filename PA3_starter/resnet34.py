import torch.nn as nn
from torchvision import transforms, models

#ToDO Fill in the __ values
class TransferLearningResNet34(nn.Module):

    def __init__(self, n_class):
        super().__init__()
        self.n_class = n_class
        resnet34 = models.resnet34(pretrained=True)
        self.relu = nn.ReLU(inplace=True)
        self.resnet34 = nn.Sequential(*list(resnet34.children())[:-2])
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64,64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.classifier = nn.Conv2d(64, self.n_class, kernel_size=1)

#TODO Complete the forward pass
    def forward(self, x):
        # x1 = self.bnd1(self.relu(self.conv1(x)))
        # # Complete the forward function for the rest of the encoder
        # x2 = self.bnd2(self.relu(self.conv2(x1)))
        # x3 = self.bnd3(self.relu(self.conv3(x2)))
        # x4 = self.bnd4(self.relu(self.conv4(x3)))
        # x5 = self.bnd5(self.relu(self.conv5(x4)))

        
        encoder_out = self.resnet34(x)
        y1 = self.bn1(self.relu(self.deconv1(encoder_out)))
        y2 = self.bn2(self.relu(self.deconv2(y1)))
        y3 = self.bn3(self.relu(self.deconv3(y2)))
        y4 = self.bn4(self.relu(self.deconv4(y3)))
        y5 = self.bn5(self.relu(self.deconv5(y4)))
        # Complete the forward function for the rest of the decoder

        score = self.classifier(y5)

        return score  # size=(N, n_class, H, W)
