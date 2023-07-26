import torch.nn as nn
import torch
import torch.nn.functional as F


#-----------------------------------------------------------------------------
class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=True, activation=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.InstanceNorm2d(out_channel))
        if activation:
            layers.append(nn.ELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
class BasicConv2(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=True, activation=True, transpose=False):
        super(BasicConv2, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.InstanceNorm2d(out_channel))
        if activation:
            layers.append(nn.ReLU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)





class RB(nn.Module):
    def __init__(self, channels):
        super(RB, self).__init__()
        self.layer_1 = BasicConv(channels, channels, 3, 1)
        self.layer_2 = BasicConv(channels, channels, 3, 1)

    def forward(self, x):
        y = self.layer_1(x)
        y = self.layer_2(y)
        return F.elu(y + x)
class RB2(nn.Module):
    def __init__(self, channels):
        super(RB2, self).__init__()
        self.layer_1 = BasicConv2(channels, channels, 3, 1)
        self.layer_2 = BasicConv2(channels, channels, 3, 1)

    def forward(self, x):
        y = self.layer_1(x)
        y = self.layer_2(y)
        return F.relu(y + x)

class Down_scale(nn.Module):
    def __init__(self, in_channel):
        super(Down_scale, self).__init__()
        self.main = BasicConv2(in_channel, in_channel*2, 3, 2)

    def forward(self, x):
        return self.main(x)

class Up_scale(nn.Module):
    def __init__(self, in_channel):
        super(Up_scale, self).__init__()
        self.main = BasicConv2(in_channel, in_channel//2, kernel_size=4, activation=True, stride=2, transpose=True)

    def forward(self, x):
        return self.main(x)

#----------FE块---------------------------------------------
class FE(nn.Module):
    def __init__(self,
                 in_channels, out_channels1, out_channels2, out_channels3, out_channels4,
                 ksize=3, stride=1, pad=1):

        super(FE, self).__init__()
        self.body1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels1, 1, 1, 0),
            nn.ReLU(inplace=True))

        self.body2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels2, ksize, stride, pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels2, out_channels2, ksize, stride, pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels2, out_channels2, ksize, stride, pad),
            nn.ReLU(inplace=True))


        self.body3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels3, ksize, stride, pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels3, out_channels3, ksize, stride, pad),
            nn.ReLU(inplace=True)
        )

        self.body4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels4, ksize, stride, pad),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out1 = self.body1(x)

        out2 = self.body2(x)

        out3 = self.body3(x)

        out4 = self.body4(x)

        out = torch.cat([out1, out2, out3, out4], dim=1)

        return out

class OneBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OneBlock, self).__init__()
        self.forw = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        x = self.forw(x)
        return x
class Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        self.forw = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels,momentum=0.9,eps=1e-04),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.forw(x)
        return x

class EBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(EBlock, self).__init__()
        self.forw = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels,momentum=0.9,eps=1e-04),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        x = self.forw(x)
        return x

class Down_Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Down_Block, self).__init__()
        self.forw = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.forw(x)
        return x

class Up_Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Up_Block, self).__init__()
        self.forw = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0,bias=True),
            nn.BatchNorm2d(out_channels,momentum=0.9,eps=1e-04),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.forw(x)
        return x
class Encoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Encoder, self).__init__()
        self.en1 = EBlock(in_channel, in_channel)
        self.en2 = EBlock(in_channel, in_channel)
        self.en3 = Down_Block(in_channel, out_channel)

    def forward(self, x):
        e1 = self.en1(x)
        e2 = self.en2(e1)
        e3 = self.en3(e2)

        return e3
class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Decoder, self).__init__()
        self.en1 = Up_Block(in_channel, in_channel)
        self.en2 = EBlock(in_channel, in_channel)
        self.en3 = EBlock(in_channel, out_channel)

    def forward(self, x):
        e1 = self.en1(x)
        e2 = self.en2(e1)
        e3 = self.en3(e2)

        return e3

#---------------------------------------------------------------------------------------------

class GradiNet(nn.Module):
    def __init__(self):
        super(GradiNet, self).__init__()
        self.one = OneBlock(128, 64)

        self.en1 = Encoder(64, 64)
        self.en2 = Encoder(64, 128)
        self.en3 = Encoder(128, 256)
        self.en4 = Encoder(256, 512)
        # --------------------------------------------
        self.mid = Block(512, 1024)
        self.mid1 = Block(1024, 512 )
        # -------------------------------------------
        self.de4 = Decoder(512, 256)
        self.de3 = Decoder(256, 128)
        self.de2 = Decoder(128, 64)
        self.de1 = Decoder(64, 64)
        self.sp = nn.Conv2d(64, 1, 3, 1, 1)
    def forward(self,input):
        head = self.one(input)
        en1 = self.en1(head)
        en2 = self.en2(en1)
        en3 = self.en3(en2)
        en4 = self.en4(en3)

        mid = self.mid(en4)
        mid1 = self.mid1(mid)

        de4 = self.de4(mid1 + en4)
        de3 = self.de3(de4 + en3)
        de2 = self.de2(de3 + en2)
        de1 = self.de1(de2 + en1)
        sp = self.sp(de1)
        return sp


class GradiNet_2(nn.Module):
    def __init__(self):
        super(GradiNet_2, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.ELU()
        )
        #
        self.downsample = nn.Sequential(
            nn.Conv2d(32,32,kernel_size=32//12,stride=32//12),
            nn.ELU()
        )
        #
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ELU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ELU()
        )
        #
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=32 // 12, stride=32 // 12),
            nn.ELU()
        )
        #
        self.conv = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
        )
    def forward(self,x):
        x = self.conv0(x)
        for j in range(5):
            resx = x
            x = F.elu(self.res_conv1(x) + resx)

        sp = self.conv(x)

        return sp


class MainNet(nn.Module):
    def __init__(self,depth=[2, 2, 2, 2]):
        super(MainNet, self).__init__()
        base_channel = 32

        # encoder
        self.Encoder = nn.ModuleList([
            BasicConv2(base_channel, base_channel, 3, 1),
            nn.Sequential(*[RB2(base_channel) for _ in range(depth[0])]),
            Down_scale(base_channel),
            BasicConv2(base_channel * 2, base_channel * 2, 3, 1),
            nn.Sequential(*[RB2(base_channel * 2) for _ in range(depth[1])]),
            Down_scale(base_channel * 2),
            BasicConv2(base_channel * 4, base_channel * 4, 3, 1),
            nn.Sequential(*[RB2(base_channel * 4) for _ in range(depth[2])]),
            Down_scale(base_channel * 4),
        ])

        self.middle = nn.Sequential(*[RB2(base_channel * 8) for _ in range(depth[3])])


        self.Decoder = nn.ModuleList([
            Up_scale(base_channel * 8),
            BasicConv2(base_channel * 8, base_channel * 4, 3, 1),
            nn.Sequential(*[RB2(base_channel * 4) for _ in range(depth[2])]),
            Up_scale(base_channel * 4),
            BasicConv2(base_channel * 4, base_channel * 2, 3, 1),
            nn.Sequential(*[RB2(base_channel * 2) for _ in range(depth[1])]),
            Up_scale(base_channel * 2),
            BasicConv2(base_channel * 2, base_channel, 3, 1),
            nn.Sequential(*[RB2(base_channel) for _ in range(depth[0])]),
        ])

        self.conv_first = BasicConv2(65, base_channel, 3, 1)
        self.conv_last = nn.Conv2d(base_channel+1, 3, 3, 1, 1)


    def encoder_1(self, x):
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if (i + 2) % 3 == 0:
                shortcuts.append(x)
        return x, shortcuts
    def decoder_1(self, x, shortcuts):
        for i in range(len(self.Decoder)):
            if (i + 2) % 3 == 0:
                index = len(shortcuts) - (i//3 + 1)
                x = torch.cat([x, shortcuts[index]], 1)
            x = self.Decoder[i](x)
        return x

    def forward(self,img_low,grad):
        x = torch.cat([img_low, grad], 1)
        x = self.conv_first(x)
        x, shortcuts = self.encoder_1(x)
        x = self.middle(x)

        x = self.decoder_1(x, shortcuts)
        x = torch.cat([x,grad],1)
        x = self.conv_last(x)
        img_color = (torch.tanh(x) + 1) / 2
        return img_color


class Gradi_ContrativeNet(nn.Module):
    def __init__(self):
        super(Gradi_ContrativeNet, self).__init__()
        self.fe =FE(3, 32, 32, 32, 32) #进3 出 32*4
        self.g_net = GradiNet_2()

        self.head1 = Block(128,64)
        self.M_net = MainNet()
        self.weight = nn.Conv2d(1, 1, 1, 1, 0, bias=False)
        self.weight1 = nn.Conv2d(1, 1, 1, 1, 0, bias=False)
    def forward(self,input):
        x= self.fe(input)
        sp = self.g_net(x)
        mask = torch.sigmoid(10 * self.weight(sp.detach())) - self.weight1(torch.sigmoid(sp.detach() + 10))
        head1 = self.head1(x)
        img_enhance = self.M_net(head1,mask)

        return img_enhance,sp


if __name__ == "__main__":

    net = Gradi_ContrativeNet()

    from torch.autograd import Variable
    inputs = Variable(torch.zeros(1, 3, 256, 256))

    outputs= net(inputs)
