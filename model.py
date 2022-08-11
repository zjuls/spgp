from layer import *

class MNIST_Net(nn.Module):  # Example net for CIFAR10
    def __init__(self, num_classes = 10, in_channels = 2, trace = False, basic_channel = 64, spgp = Flase):
        super(MNIST_Net, self).__init__()
        self.basic_channel = basic_channel
        self.trace = trace
        self.spgp = spgp
        self.bn0 = TemporalBN(self.basic_channel,steps)#tdBatchNorm(nn.BatchNorm2d(64))
        self.bn1 = TemporalBN(self.basic_channel*2,steps)#tdBatchNorm(nn.BatchNorm2d(128))

        self.conv0_s = tdLayer(nn.Conv2d(in_channels, self.basic_channel, 3, 1, 1, bias=True))
        self.pool0_s = tdLayer(nn.AvgPool2d(2))
        self.conv1_s = tdLayer(nn.Conv2d(self.basic_channel, self.basic_channel*2, 3, 1, 1, bias=True))
        if not self.spgp:
            self.pool1_s = tdLayer(nn.AvgPool2d(2))
            self.fc1_s = tdLayer(nn.Linear(7 * 7 * self.basic_channel * 2, 200, bias=True))
            self.fc2_s = tdLayer(nn.Linear(200, num_classes, bias=True))
        else:
            self.pool1_s = SPGP()
            self.fc1_s = tdLayer(nn.Linear(self.basic_channel * 2, num_classes, bias=True))
        self.spike = LIF()
    
        self.trace = trace
    def forward(self, x):
        x, _ = torch.broadcast_tensors(x, torch.zeros((steps,) + x.shape))
        x = x.permute(1, 2, 3, 4, 0)

        x = self.bn0(self.conv0_s(x.float()))
        x = self.spike(x)
        x = self.pool0_s(x)
        

        x = self.bn1(self.conv1_s(x))
        x = self.spike(x)
        if not self.spgp
            x = self.pool1_s(x)
            x = x.view(x.shape[0], -1, x.shape[4])
            x = self.fc1_s(x)
            x = self.spike(x)
            x = self.fc2_s(x)
            x = self.spike(x, output = True, vmem = self.trace)
        else:
            x = self.pool1_s(x)
            x = x.view(x.shape[0], -1, x.shape[4])
            x = self.fc1_s(x)
            x = self.spike(x, output = True, vmem = self.trace)
        return x


if __name__ == '__main__':
    x = torch.ones((1, 1, 28, 28), dtype=torch.float32, device = device)
    snn = MNIST_Net(10, 1, True, 64)#ResNet_small(BasicBlock, [2, 2, 2, 2], 10)
    snn.to(device)
    print(snn(x))
