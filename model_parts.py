import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, leakyness=0):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(negative_slope=leakyness, inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=leakyness, inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
#%%
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)
class UpConv(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels=16):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1_up = self.up(x1)
        x_concat = torch.cat([x1_up, x2], axis=1)
        x = self.conv(x_concat)
        return x
#%%
class UnetDownwards(nn.Module):
    def __init__(self,num_levels=2 ,init_channels=3, first_output = 32) -> None:
        self.num_levels = num_levels
        super().__init__()
        self.first_dc = DoubleConv(in_channels=init_channels, out_channels= first_output)
        num_channels = first_output
        for num in range(num_levels):
            setattr(self, f'down_{num}', Down(in_channels=num_channels, out_channels= num_channels*2))
            num_channels*=2
    def forward(self, x):
        x_out = self.first_dc(x)
        outputs =[x_out]
        for num in range(self.num_levels):
            x_out = getattr(self, f'down_{num}')(x_out)
            outputs.append(x_out)
        return outputs

#%%
class UnetUpwards(nn.Module):
    def __init__(self,num_levels=2 ,output_channels=2, first_output = 32) -> None:
        self.num_levels = num_levels
        super().__init__()
        num_channels = first_output * 2**num_levels
        for num in range(num_levels):
            setattr(self, f'up_{num}', UpConv(in_channels=num_channels, out_channels= num_channels//2))
            num_channels//=2
        self.output_dc = DoubleConv(in_channels=num_channels, out_channels= output_channels, mid_channels=num_channels*2, leakyness=0)
    def forward(self, x):
        output = x[-1]
        for num in range(self.num_levels):
            output = getattr(self, f'up_{num}')(output, x[self.num_levels - num - 1] )
        output = self.output_dc(output)
        return output
#%%
class UnetTop(nn.Module):
    def __init__(self ,output_channels=2, first_output = 32) -> None:
        super().__init__()
        self.conv_1 = DoubleConv(in_channels=first_output, out_channels= first_output*2, leakyness=0.1)
        self.conv_2 = DoubleConv(in_channels=first_output*2, out_channels= output_channels, leakyness=0.1, mid_channels=first_output)

    def forward(self, x):
        output = self.conv_1(x)
        output = self.conv_2(output)

        return output
#%%
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, instance_norm = True):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.instance_norm = instance_norm
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        if self.instance_norm:
            self.inst_norm = nn.InstanceNorm2d(out_channels, affine=True)
        

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.instance_norm:
            out = self.inst_norm(out)
        return out

class SquuezeExcitation(torch.nn.Module):
    def __init__(self, channels, reduction_factor=4):
        super(SquuezeExcitation, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.squeeze = nn.Linear(in_features=channels, out_features=channels//reduction_factor)
        self.excitation = nn.Linear(in_features=channels//reduction_factor, out_features=channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.global_pool(x)
        out = self.squeeze(out.permute((0,2,3,1)))
        out = self.relu(out)
        out = self.excitation(out)
        out = self.sigmoid(out)
        return out.permute((0,3,1,2))

class ResidualBlock(torch.nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels, kernel_size=3, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=kernel_size, stride=stride)
        self.conv2 = ConvLayer(channels, channels, kernel_size=kernel_size, stride=stride)
        self.sqexc = SquuezeExcitation(channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        sande = self.sqexc(out)
        out = out*sande + residual
        return out


class UpsampleConvLayer(torch.nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None, instance_norm=True):
        super(UpsampleConvLayer, self).__init__()
        self.instance_norm = instance_norm
        self.upsample = upsample
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        if self.instance_norm:
            self.inst_norm = nn.InstanceNorm2d(out_channels, affine=True)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = nn.functional.interpolate(x_in, mode='nearest', scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        if self.instance_norm:
            out = self.inst_norm(out)
        return out
#%%
class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim,activation):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1) #
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize,C,width ,height = x.size()
        proj_query  = self.query_conv(x).view(m_batchsize,-1,width*height).permute(0,2,1) # B X CX(N)
        proj_key =  self.key_conv(x).view(m_batchsize,-1,width*height) # B X C x (*W*H)
        energy =  torch.bmm(proj_query,proj_key) # transpose check
        attention = self.softmax(energy) # BX (N) X (N) 
        proj_value = self.value_conv(x).view(m_batchsize,-1,width*height) # B X C X N

        out = torch.bmm(proj_value,attention.permute(0,2,1) )
        out = out.view(m_batchsize,C,width,height)
        
        out = self.gamma*out + x
        return out,attention
#%%
class UpDownResBlock(nn.Module):
    def __init__(self, channels):
        super(UpDownResBlock, self).__init__()
        self.res_up = ResidualBlock(channels)
        self.res_down = ResidualBlock(channels)
        self.down = Down(channels, channels)
        self.up = UpsampleConvLayer(channels, channels, kernel_size=3, stride=1, upsample=2)
    def forward(self, x):
        # x.shape = (B, C, H, W)
        up = self.res_up(x) # (B, C, H, W)
        down = self.down(x) # (B, C, H/2, W/2)
        down = self.res_down(down) # (B, C, H/2, W/2)
        down = self.up(down) # (B, C, H, W)
        out = up + down # (B, C, H, W)
        return out

class MaxAvgPool2d(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(MaxAvgPool2d, self).__init__()
        self.maxpool = nn.MaxPool2d(**kwargs)
        self.avgpool = nn.AvgPool2d(**kwargs)
    def forward(self,x):
        maxp = self.maxpool(x)
        avgp = self.avgpool(x)
        return (maxp + avgp) /2

class MaxAvgPoolAdaptative2d(nn.Module):
    def __init__(self, output_size) -> None:
        super(MaxAvgPoolAdaptative2d, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=output_size)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=output_size)
    def forward(self,x):
        maxp = self.maxpool(x)
        avgp = self.avgpool(x)
        return (maxp + avgp) /2