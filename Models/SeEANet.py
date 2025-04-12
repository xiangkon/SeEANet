import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary

class InceptionBlk(nn.Module):
    def __init__(self, inch, outch):
        super(InceptionBlk, self).__init__()
        self.b1_conv1 = nn.Conv1d(inch, outch, kernel_size=1)
        self.b1_lrelu = nn.LeakyReLU(negative_slope=0.3)
        
        self.b2_conv1 = nn.Conv1d(inch, outch, kernel_size=1)
        self.b2_lrelu1 = nn.LeakyReLU(negative_slope=0.3)
        self.b2_conv2 = nn.Conv1d(outch, outch, kernel_size=5, padding=2)
        self.b2_lrelu2 = nn.LeakyReLU(negative_slope=0.3)
        
        self.b3_conv1 = nn.Conv1d(inch, outch, kernel_size=1)
        self.b3_lrelu1 = nn.LeakyReLU(negative_slope=0.3)
        self.b3_conv2 = nn.Conv1d(outch, outch, kernel_size=9, padding=4)
        self.b3_lrelu2 = nn.LeakyReLU(negative_slope=0.3)
        
        self.b4_mpool = nn.MaxPool1d(3, stride=1, padding=1)
        self.b4_conv1 = nn.Conv1d(inch, outch, kernel_size=1)
    
    def forward(self, x):
        x1 = self.b1_conv1(x)
        x1 = self.b1_lrelu(x1)

        x2 = self.b2_conv1(x)
        x2 = self.b2_lrelu1(x2)
        x2 = self.b2_conv2(x2)
        x2 = self.b2_lrelu2(x2)

        x3 = self.b3_conv1(x)
        x3 = self.b3_lrelu1(x3)
        x3 = self.b3_conv2(x3)
        x3 = self.b3_lrelu2(x3)

        x4 = self.b4_mpool(x)
        x4 = self.b4_conv1(x4)

        x = torch.cat([x1, x2, x3, x4], dim=1)

        return x


class HardSwish(nn.Module):
    def __init__(self):
        super(HardSwish, self).__init__()
    
    def forward(self, x):
        return x * torch.clamp(F.relu(x + 3.0), max=6.0) / 6.0
    
class FirstConvBlk(nn.Module):
        def __init__(self, outch, input_shape = [256, 6]):
            
            super(FirstConvBlk, self).__init__()        
            # Shallow feature extraction module
            self.first_conv_blocks = nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(1, outch, kernel_size=9, padding=4),
                    nn.LeakyReLU(negative_slope=0.3),
                    nn.MaxPool1d(4),
                    nn.Dropout(0.2)
                ) for _ in range(input_shape[1])
                ])
        def forward(self, x):
            first_conv_outs = []
            for i in range(x.shape[1]):
                x_input = x[:, i:i+1, :]  # Slice operation
                out = self.first_conv_blocks[i](x_input)
                first_conv_outs.append(out)
        
            x = torch.cat(first_conv_outs, dim=1)
            return x
        
class ChannelSpilit(nn.Module):
    def __init__(self, groups=4):
        super(ChannelSpilit, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch, filters, width = x.shape
        channels_per_group = filters // self.groups
        x = x.view(-1, channels_per_group, width)
        x1, x2 = torch.split(x, x.size(1) // 2, dim=1)
        return x1, x2

class ChannelShuffler(torch.nn.Module):
    def __init__(self, num, groups=2):
        super(ChannelShuffler, self).__init__()
        self.num = num
        self.groups = groups

    def forward(self, x):
        b, c, w = x.shape
        x = x.reshape(b, self.groups, -1, w)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(b, -1, w)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        self.fc = nn.Linear(channels, channels, bias=True)
        self.fc.weight.data.zero_()
        self.fc.bias.data.fill_(1)
        self.activation = nn.Hardsigmoid()

    def forward(self, x):
        batch_size, channels, width = x.shape
        x_global_avg_pool = torch.mean(x, dim=2)
        y = self.fc(x_global_avg_pool)
        y = self.activation(y)
        y = y.view(batch_size, channels, 1)
        output = x * y
        return output


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        self.group_norm = nn.GroupNorm(channels, channels)
        self.fc = nn.Linear(64, 64, bias=True)
        self.activation = nn.Hardsigmoid()

    def forward(self, x):
        x_grop_norm = self.group_norm(x)
        x_global_avg_pool = torch.mean(x_grop_norm, dim=1)
        y = self.fc(x_global_avg_pool)
        y = self.activation(y)
        y = y.view(y.size(0), 1, y.size(1))
        output = x * y
        return output


class SABlk(nn.Module):
    def __init__(self, groups, num):
        super(SABlk, self).__init__()
        self.num = num
        self.channel_spilit = ChannelSpilit(groups)
        self.cam = ChannelAttention(8)
        self.sam = SpatialAttention(8)
        self.channel_shuffle = ChannelShuffler(num, 2)

    def forward(self, x):
        channel, len = x.shape[1:]
        x1, x2 = self.channel_spilit(x)
        x1 = self.cam(x1)
        x2 = self.sam(x2)
        y = torch.cat([x1, x2], dim=1)
        y = y.view(-1, channel, len)
        y = self.channel_shuffle(y)

        return y

class ImprovedBottleneck(nn.Module):
    def __init__(self, ImproverFilter, kernel, e, num=1, groups=8, alpha=1.0):
        super(ImprovedBottleneck, self).__init__()
        cchannel = int(alpha * ImproverFilter)

        self.conv_block = nn.Sequential(
            nn.LazyConv1d(e, kernel_size=1),
            nn.BatchNorm1d(e),
            HardSwish()
        )
        self.depthwise_conv = nn.LazyConv1d(out_channels=128, kernel_size=kernel, stride=1, padding=4, groups=128)
        self.bn1 = nn.BatchNorm1d(128)
        self.hs1 = HardSwish()
        self.sa = SABlk(groups, num)
        self.conv1 = nn.LazyConv1d(cchannel, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(num_features=cchannel)

        

    def forward(self, x):
        init_filters = x.shape[1]
        y = self.conv_block(x)
        y = self.depthwise_conv(y)
        y = self.bn1(y)
        y = self.hs1(y)
        y = self.sa(y)
        y = self.conv1(y)
        y = self.bn2(y)

        if y.shape[1] == init_filters:
            y += x

        return y

class SeEANet(nn.Module):
    def __init__(self, PreLen=1, PreNum=1, input_shape = [256, 6]):
        super(SeEANet, self).__init__()
        firstFilter = 8
        inceptionFilter = input_shape[1] * firstFilter
        ImproverFilter = 4 * inceptionFilter

        self.FirConv = FirstConvBlk(outch=firstFilter, input_shape=input_shape)
        self.InceptionBlk = InceptionBlk(inch=inceptionFilter, outch=inceptionFilter)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.3)
        self.atten1 = ImprovedBottleneck(ImproverFilter=ImproverFilter, kernel=9, e=128, groups=8, alpha=1.0)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.3)
        self.mpool1 = nn.MaxPool1d(kernel_size=3)
        self.dp1 = nn.Dropout1d(0.2)
        self.conv1 = nn.LazyConv1d(out_channels=64, kernel_size=1)

        self.lstm1 = nn.LSTM(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.dp2 = nn.Dropout1d(0.2)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.dp3 = nn.Dropout1d(0.2)
        self.l = nn.LazyLinear(PreLen*PreNum)
        

    def forward(self, x):
        x = self.FirConv(x)
        x = self.InceptionBlk(x)
        x = self.lrelu1(x)
        x = self.atten1(x)
        x = self.lrelu2(x)
        x = self.mpool1(x)
        x = self.dp1(x)
        x = self.conv1(x)

        x = x.permute(0, 2, 1)
        x,_ = self.lstm1(x)
        x = self.dp2(x)
        x,_ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.dp3(x)
        y = self.l(x)


        return y
    
if __name__ == '__main__':
    model = SeEANet()
    model = model.cuda()
    inputs = torch.randn(24, 6, 256)
    inputs = inputs.cuda()
    # summary(model, input_size=(6, 256))
    outputs = model(inputs)
    print("outpus's shape :", outputs.shape)
