import torch.nn as nn

class labelDiscriminator_seg(nn.Module):
    def __init__(self, num_classes, ndf=64):
        super(labelDiscriminator_seg, self).__init__()
        # self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2,padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2,padding=1)
        self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2,padding=1)
        self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2,padding=1)
        self.classifier = nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=2,padding=1)
        self.gpN1 = nn.GroupNorm(num_groups=32, num_channels=ndf, eps=1e-5, affine=False)
        self.gpN2 = nn.GroupNorm(num_groups=32, num_channels=ndf * 2, eps=1e-5, affine=False)
        self.gpN3 = nn.GroupNorm(num_groups=32, num_channels=ndf * 3, eps=1e-5, affine=False)
        self.gpN4 = nn.GroupNorm(num_groups=32, num_channels=ndf * 4, eps=1e-5, affine=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gpN1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.gpN2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.gpN3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.gpN4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)

        return x
