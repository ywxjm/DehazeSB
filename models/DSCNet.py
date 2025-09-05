
import torch
from torch import nn, cat
from torch.nn.functional import dropout
from models.DSConv import DSConv, DSConv_pro
# from models.basicsr.archs.rrdbnet_arch import RRDB
from models.ncsn_networks import PixelNorm, get_timestep_embedding,Downsample,Upsample


class InceptionDWConv2d(nn.Module):
    """ Inception depthweise convolution
    """

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125):
        super().__init__()

        gc = int(in_channels * branch_ratio)  # channel numbers of a convolution branch
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)),
            dim=1,
        )

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.ReLU,
        drop=0.0,
        mlp_ratio=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        # hidden_features = hidden_features or in_features
        hidden_features = int(in_features * mlp_ratio)
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EncoderConvInception(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(EncoderConvInception, self).__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.conv_1x1 = nn.Conv2d(in_ch, in_ch, 1, padding=0)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.inception_conv = InceptionDWConv2d(in_ch)
        self.conv_out = nn.Conv2d(in_ch * 3, out_ch, 3, padding=1)
        # self.mlp = Mlp(in_features=in_ch * 3, mlp_ratio=1, out_features=out_ch)

    def forward(self, x):
        x_3x3 = self.conv(x)
        x_1x1 = self.conv_1x1(x)
        x_ir = self.inception_conv(x)
        x_ = torch.cat([x_1x1, x_3x3, x_ir], dim=1)
        x_ =  self.relu(x_)
        x = self.conv_out(x_)
        x = self.gn(x)
        x = self.relu(x)
        return x



# Define a standard convolution kernel
class EncoderConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(EncoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x


class DecoderConvInception(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DecoderConvInception, self).__init__()
        self.conv = nn.Conv2d(in_ch, in_ch, 3, padding=1)
        self.conv_1x1 = nn.Conv2d(in_ch, in_ch, 1, padding=0)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.inception_conv = InceptionDWConv2d(in_ch)
        self.mlp = Mlp(in_features=in_ch * 3, mlp_ratio=1, out_features=out_ch)

    def forward(self, x):
        x_3x3 = self.conv(x)
        x_1x1 = self.conv_1x1(x)
        x_ir = self.inception_conv(x)
        x = self.mlp(torch.cat([x_1x1, x_3x3, x_ir], dim=1))
        x = self.gn(x)
        x = self.relu(x)
        return x

class DecoderConv(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(DecoderConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn = nn.GroupNorm(out_ch // 4, out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)

        return x
### DSCNet_original 含有DSC卷积
class DSCNet_pro(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
        kernel_size,
        extend_scope,
        if_offset,
        device,
        number,
        dim,
        n_blocks,
        padding_type,
        use_dropout
    ):
        """
        Our DSCNet
        :param n_channels: input channel
        :param n_classes: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        :param device: set on gpu
        :param number: basic layer numbers
        :param dim:
        """
        super().__init__()
        device = device
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.relu = nn.ReLU(inplace=True)
        self.number = number
        """
        The three contributions proposed in our paper are relatively independent. 
        In order to facilitate everyone to use them separately, 
        we first open source the network part of DSCNet. 
        <dim> is a parameter used by multiple templates, 
        which we will open source in the future ...
        """
        self.dim = dim  # This version dim is set to 1 by default, referring to a group of x-axes and y-axes
        """
        Here is our framework. Since the target also has non-tubular structure regions, 
        our designed model also incorporates the standard convolution kernel, 
        for fairness, we also add this operation to compare with other methods (like: Deformable Convolution).
        """
        self.conv00 = EncoderConv(n_channels, self.number)
        self.conv0x = DSConv_pro(
            n_channels,
            self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv0y = DSConv_pro(
            n_channels,
            self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv1 = EncoderConv(3 * self.number, self.number)

        self.conv20 = EncoderConv(self.number, 2 * self.number)
        self.conv2x = DSConv_pro(
            self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv2y = DSConv_pro(
            self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv3 = EncoderConv(6 * self.number, 2 * self.number)

        self.conv40 = EncoderConv(2 * self.number, 4 * self.number)
        self.conv4x = DSConv_pro(
            2 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv4y = DSConv_pro(
            2 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv5 = EncoderConv(12 * self.number, 4 * self.number)

        self.conv60 = EncoderConv(4 * self.number, 8 * self.number)
        self.conv6x = DSConv_pro(
            4 * self.number,
            8 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv6y = DSConv_pro(
            4 * self.number,
            8 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv7 = EncoderConv(24 * self.number, 8 * self.number)

        self.conv120 = EncoderConv(12 * self.number, 4 * self.number)
        self.conv12x = DSConv_pro(
            12 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv12y = DSConv_pro(
            12 * self.number,
            4 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv13 = EncoderConv(12 * self.number, 4 * self.number)

        self.conv140 = DecoderConv(6 * self.number, 2 * self.number)
        self.conv14x = DSConv_pro(
            6 * self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv14y = DSConv_pro(
            6 * self.number,
            2 * self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv15 = DecoderConv(6 * self.number, 2 * self.number)

        self.conv160 = DecoderConv(3 * self.number, self.number)
        self.conv16x = DSConv_pro(
            3 * self.number,
            self.number,
            self.kernel_size,
            self.extend_scope,
            0,
            self.if_offset,
            device,
        )
        self.conv16y = DSConv_pro(
            3 * self.number,
            self.number,
            self.kernel_size,
            self.extend_scope,
            1,
            self.if_offset,
            device,
        )
        self.conv17 = DecoderConv(3 * self.number, self.number)

        self.out_conv = nn.Conv2d(self.number, n_classes, 1)
        ###
        self.sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)
        ###
        ###
        self.z_transform = ZTransform(16 * 16, 8 * self.number, n_out=2)  
        self.time_embed = TimeEmbedding(4 *self.number, 8 * self.number) 
        self.model_res = nn.ModuleList()
        for i in range(n_blocks):
            self.model_res += [ResnetBlock_cond(8 * self.number, padding_type=padding_type, use_dropout=use_dropout, temb_dim=8 * self.number,z_dim=8 * self.number)]

    def forward(self, x,time_cond,z,layers=[], encode_only=False):
        ### first embed noise and time information
        z_embed = self.z_transform(z)  
        temb = get_timestep_embedding(time_cond, self.number*4)  
        time_embed = self.time_embed(temb) 
        if len(layers) > 0:
            feats = []
        # block0
            x_00_0 = self.conv00(x)
            x_0x_0 = self.conv0x(x)
            x_0y_0 = self.conv0y(x)
            x_0_1 = self.conv1(torch.cat([x_00_0, x_0x_0, x_0y_0], dim=1))
            feats.append(x_0_1)

        # block1
            x = Downsample(x_0_1.shape[1])(x_0_1)
            x_20_0 = self.conv20(x)
            x_2x_0 = self.conv2x(x)
            x_2y_0 = self.conv2y(x)
            x_1_1 = self.conv3(torch.cat([x_20_0, x_2x_0, x_2y_0], dim=1))
            feats.append(x_1_1)

        # block2
            x = Downsample(x_1_1.shape[1])(x_1_1)
            x_40_0 = self.conv40(x)
            x_4x_0 = self.conv4x(x)
            x_4y_0 = self.conv4y(x)
            x_2_1 = self.conv5(torch.cat([x_40_0, x_4x_0, x_4y_0], dim=1))
            feats.append(x_2_1)

        # block3
            x = Downsample(x_2_1.shape[1])(x_2_1)
            x_60_0 = self.conv60(x)
            x_6x_0 = self.conv6x(x)
            x_6y_0 = self.conv6y(x)
            x_3_1 = self.conv7(torch.cat([x_60_0, x_6x_0, x_6y_0], dim=1)) 
            feats.append(x_3_1)

        ### add middle resnet block to encorporate time and z information
            for layer_id, layer in enumerate(self.model_res):  
                x_3_1 = layer(x_3_1,time_embed,z_embed)   
                if layer_id in layers:
                    feats.append(x_3_1)
                if layer_id == layers[-1] and encode_only:
                    return feats
            return x_3_1, feats
            
        else:
            x_00_0 = self.conv00(x)
            x_0x_0 = self.conv0x(x)
            x_0y_0 = self.conv0y(x)
            x_0_1 = self.conv1(torch.cat([x_00_0, x_0x_0, x_0y_0], dim=1))
            ##copy feature

        # block1
            x = Downsample(x_0_1.shape[1])(x_0_1)
            x_20_0 = self.conv20(x)
            x_2x_0 = self.conv2x(x)
            x_2y_0 = self.conv2y(x)
            x_1_1 = self.conv3(torch.cat([x_20_0, x_2x_0, x_2y_0], dim=1))

        # block2
            x = Downsample(x_1_1.shape[1])(x_1_1)
            x_40_0 = self.conv40(x)
            x_4x_0 = self.conv4x(x)
            x_4y_0 = self.conv4y(x)
            x_2_1 = self.conv5(torch.cat([x_40_0, x_4x_0, x_4y_0], dim=1))

        # block3
            x = Downsample(x_2_1.shape[1])(x_2_1)
            x_60_0 = self.conv60(x)
            x_6x_0 = self.conv6x(x)
            x_6y_0 = self.conv6y(x)
            x_3_1 = self.conv7(torch.cat([x_60_0, x_6x_0, x_6y_0], dim=1)) 

            for layer_id, layer in enumerate(self.model_res):  
                x_3_1 = layer(x_3_1,time_embed,z_embed) 

            # block4
            x = Upsample(x_3_1.shape[1])(x_3_1)
            x_120_2 = self.conv120(torch.cat([x, x_2_1], dim=1)) 
            x_12x_2 = self.conv12x(torch.cat([x, x_2_1], dim=1))
            x_12y_2 = self.conv12y(torch.cat([x, x_2_1], dim=1))
            x_2_3 = self.conv13(torch.cat([x_120_2, x_12x_2, x_12y_2], dim=1))

            # block5
            x = Upsample(x_2_3.shape[1])(x_2_3)
            x_140_2 = self.conv140(torch.cat([x, x_1_1], dim=1))
            x_14x_2 = self.conv14x(torch.cat([x, x_1_1], dim=1))
            x_14y_2 = self.conv14y(torch.cat([x, x_1_1], dim=1))
            x_1_3 = self.conv15(torch.cat([x_140_2, x_14x_2, x_14y_2], dim=1))

            # block6
            x = Upsample(x_1_3 .shape[1])(x_1_3)
            x_160_2 = self.conv160(torch.cat([x, x_0_1], dim=1))
            x_16x_2 = self.conv16x(torch.cat([x, x_0_1], dim=1))
            x_16y_2 = self.conv16y(torch.cat([x, x_0_1], dim=1))
            x_0_3 = self.conv17(torch.cat([x_160_2, x_16x_2, x_16y_2], dim=1))
            out = self.out_conv(x_0_3)
            out = self.Tanh(out)

        return out
### helper block
### DSCNet_unet，拆掉DSC卷积
class DSCNet_unet(nn.Module):
    def __init__(
            self,
            n_channels,
            n_classes,
            kernel_size,
            extend_scope,
            if_offset,
            device,
            number,
            dim,
            n_blocks,
            padding_type,
            use_dropout
    ):
        """
        Our DSCNet
        :param n_channels: input channel
        :param n_classes: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        :param device: set on gpu
        :param number: basic layer numbers
        :param dim:
        """
        super().__init__()
        device = device
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.relu = nn.ReLU(inplace=True)
        self.number = number
        """
        The three contributions proposed in our paper are relatively independent. 
        In order to facilitate everyone to use them separately, 
        we first open source the network part of DSCNet. 
        <dim> is a parameter used by multiple templates, 
        which we will open source in the future ...
        """
        self.dim = dim  # This version dim is set to 1 by default, referring to a group of x-axes and y-axes
        """
        Here is our framework. Since the target also has non-tubular structure regions, 
        our designed model also incorporates the standard convolution kernel, 
        for fairness, we also add this operation to compare with other methods (like: Deformable Convolution).
        """
        self.conv00 = EncoderConv(n_channels, self.number)
        self.conv1 = EncoderConv(1 * self.number, self.number)
        self.conv20 = EncoderConv(self.number, 2 * self.number)
        self.conv3 = EncoderConv(2 * self.number, 2 * self.number)

        self.conv40 = EncoderConv(2 * self.number, 4 * self.number)
        self.conv5 = EncoderConv(4 * self.number, 4 * self.number)

        self.conv60 = EncoderConv(4 * self.number, 8 * self.number)
        self.conv7 = EncoderConv(8 * self.number, 8 * self.number)

        self.conv120 = EncoderConv(12 * self.number, 4 * self.number)
        self.conv13 = EncoderConv(4 * self.number, 4 * self.number)

        self.conv140 = DecoderConv(6 * self.number, 2 * self.number)
        self.conv15 = DecoderConv(2 * self.number, 2 * self.number)

        self.conv160 = DecoderConv(3 * self.number, self.number)
        self.conv17 = DecoderConv(1 * self.number, self.number)
        self.out_conv = nn.Conv2d(self.number, n_classes, 1)
        ###
        self.sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)
        ###
        ###
        self.z_transform = ZTransform(16 * 16, 8 * self.number, n_out=2)
        self.time_embed = TimeEmbedding(4 * self.number, 8 * self.number)
        self.model_res = nn.ModuleList()
        for i in range(n_blocks):
            self.model_res += [ResnetBlock_cond(8 * self.number, padding_type=padding_type, use_dropout=use_dropout,
                                                temb_dim=8 * self.number, z_dim=8 * self.number)]

    def forward(self, x, time_cond, z, layers=[], encode_only=False):
        ### first embed noise and time information
        z_embed = self.z_transform(z)
        temb = get_timestep_embedding(time_cond, self.number * 4)
        time_embed = self.time_embed(temb)
        if len(layers) > 0:
            feats = []
            # block0
            x_00_0 = self.conv00(x)
            x_0_1 = self.conv1(x_00_0)
            feats.append(x_0_1)

            # block1
            x = Downsample(x_0_1.shape[1])(x_0_1)
            x_20_0 = self.conv20(x)
            x_1_1 = self.conv3(x_20_0)
            feats.append(x_1_1)

            # block2
            x = Downsample(x_1_1.shape[1])(x_1_1)
            x_40_0 = self.conv40(x)
            x_2_1 = self.conv5(x_40_0)
            feats.append(x_2_1)

            # block3
            x = Downsample(x_2_1.shape[1])(x_2_1)
            x_60_0 = self.conv60(x)
            x_3_1 = self.conv7(x_60_0)
            feats.append(x_3_1)

            ### add middle resnet block to encorporate time and z information
            for layer_id, layer in enumerate(self.model_res):
                x_3_1 = layer(x_3_1, time_embed, z_embed)
                if layer_id in layers:
                    feats.append(x_3_1)
                if layer_id == layers[-1] and encode_only:
                    return feats
            return x_3_1, feats

        else:
            x_00_0 = self.conv00(x)
            x_0_1 = self.conv1(x_00_0)
            ##copy feature

            # block1
            x = Downsample(x_0_1.shape[1])(x_0_1)
            x_20_0 = self.conv20(x)
            x_1_1 = self.conv3(x_20_0)

            # block2
            x = Downsample(x_1_1.shape[1])(x_1_1)
            x_40_0 = self.conv40(x)
            x_2_1 = self.conv5(x_40_0)

            # block3
            x = Downsample(x_2_1.shape[1])(x_2_1)
            x_60_0 = self.conv60(x)
            x_3_1 = self.conv7(x_60_0)

            for layer_id, layer in enumerate(self.model_res):
                x_3_1 = layer(x_3_1, time_embed, z_embed)

                # block4
            x = Upsample(x_3_1.shape[1])(x_3_1)
            x_120_2 = self.conv120(torch.cat([x, x_2_1], dim=1))
            x_2_3 = self.conv13(x_120_2)

            # block5
            x = Upsample(x_2_3.shape[1])(x_2_3)
            x_140_2 = self.conv140(torch.cat([x, x_1_1], dim=1))
            x_1_3 = self.conv15(x_140_2)

            # block6
            x = Upsample(x_1_3.shape[1])(x_1_3)
            x_160_2 = self.conv160(torch.cat([x, x_0_1], dim=1))
            x_0_3 = self.conv17(x_160_2)
            out = self.out_conv(x_0_3)
            out = self.Tanh(out)

        return out


class DSCNet_unetgan(nn.Module):
    def __init__(
            self,
            n_channels,
            n_classes,
            kernel_size,
            extend_scope,
            if_offset,
            device,
            number,
            dim,
            n_blocks,
            padding_type,
            use_dropout
    ):
        """
        Our DSCNet
        :param n_channels: input channel
        :param n_classes: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        :param device: set on gpu
        :param number: basic layer numbers
        :param dim:
        """
        super().__init__()
        device = device
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.relu = nn.ReLU(inplace=True)
        self.number = number
        """
        The three contributions proposed in our paper are relatively independent. 
        In order to facilitate everyone to use them separately, 
        we first open source the network part of DSCNet. 
        <dim> is a parameter used by multiple templates, 
        which we will open source in the future ...
        """
        self.dim = dim  # This version dim is set to 1 by default, referring to a group of x-axes and y-axes
        """
        Here is our framework. Since the target also has non-tubular structure regions, 
        our designed model also incorporates the standard convolution kernel, 
        for fairness, we also add this operation to compare with other methods (like: Deformable Convolution).
        """
        self.conv00 = EncoderConv(n_channels, self.number)
        self.conv1 = EncoderConv(1 * self.number, self.number)
        self.conv20 = EncoderConv(self.number, 2 * self.number)
        self.conv3 = EncoderConv(2 * self.number, 2 * self.number)

        self.conv40 = EncoderConv(2 * self.number, 4 * self.number)
        self.conv5 = EncoderConv(4 * self.number, 4 * self.number)

        self.conv60 = EncoderConv(4 * self.number, 8 * self.number)
        self.conv7 = EncoderConv(8 * self.number, 8 * self.number)

        self.conv120 = EncoderConv(12 * self.number, 4 * self.number)
        self.conv13 = EncoderConv(4 * self.number, 4 * self.number)

        self.conv140 = DecoderConv(6 * self.number, 2 * self.number)
        self.conv15 = DecoderConv(2 * self.number, 2 * self.number)

        self.conv160 = DecoderConv(3 * self.number, self.number)
        self.conv17 = DecoderConv(1 * self.number, self.number)
        self.out_conv = nn.Conv2d(self.number, n_classes, 1)
        ###
        self.sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)
        ###
        ###
        self.z_transform = ZTransform(16 * 16, 8 * self.number, n_out=2)
        self.time_embed = TimeEmbedding(4 * self.number, 8 * self.number)
        self.model_res = nn.ModuleList()
        for i in range(n_blocks):
            self.model_res += [ResnetBlock_cond(8 * self.number, padding_type=padding_type, use_dropout=use_dropout,
                                                temb_dim=8 * self.number, z_dim=8 * self.number)]

    def forward(self, x, time_cond, z, layers=[], encode_only=False):
        ### first embed noise and time information
        z_embed = self.z_transform(z)
        temb = get_timestep_embedding(time_cond, self.number * 4)
        time_embed = self.time_embed(temb)
        if len(layers) > 0:
            feats = []
            # block0
            x_00_0 = self.conv00(x)
            x_0_1 = self.conv1(x_00_0)
            feats.append(x_0_1)

            # block1
            x = Downsample(x_0_1.shape[1])(x_0_1)
            x_20_0 = self.conv20(x)
            x_1_1 = self.conv3(x_20_0)
            feats.append(x_1_1)

            # block2
            x = Downsample(x_1_1.shape[1])(x_1_1)
            x_40_0 = self.conv40(x)
            x_2_1 = self.conv5(x_40_0)
            feats.append(x_2_1)

            # block3
            x = Downsample(x_2_1.shape[1])(x_2_1)
            x_60_0 = self.conv60(x)
            x_3_1 = self.conv7(x_60_0)
            feats.append(x_3_1)

            ### add middle resnet block to encorporate time and z information
            for layer_id, layer in enumerate(self.model_res):
                x_3_1 = layer(x_3_1, time_embed, z_embed)
                if layer_id in layers:
                    feats.append(x_3_1)
                if layer_id == layers[-1] and encode_only:
                    return feats
            return x_3_1, feats

        else:
            x_00_0 = self.conv00(x)
            x_0_1 = self.conv1(x_00_0)
            ##copy feature

            # block1
            x = Downsample(x_0_1.shape[1])(x_0_1)
            x_20_0 = self.conv20(x)
            x_1_1 = self.conv3(x_20_0)

            # block2
            x = Downsample(x_1_1.shape[1])(x_1_1)
            x_40_0 = self.conv40(x)
            x_2_1 = self.conv5(x_40_0)

            # block3
            x = Downsample(x_2_1.shape[1])(x_2_1)
            x_60_0 = self.conv60(x)
            x_3_1 = self.conv7(x_60_0)

            for layer_id, layer in enumerate(self.model_res):
                x_3_1 = layer(x_3_1, time_embed, z_embed)

                # block4
            x = Upsample(x_3_1.shape[1])(x_3_1)
            x_120_2 = self.conv120(torch.cat([x, x_2_1], dim=1))
            x_2_3 = self.conv13(x_120_2)

            # block5
            x = Upsample(x_2_3.shape[1])(x_2_3)
            x_140_2 = self.conv140(torch.cat([x, x_1_1], dim=1))
            x_1_3 = self.conv15(x_140_2)

            # block6
            x = Upsample(x_1_3.shape[1])(x_1_3)
            x_160_2 = self.conv160(torch.cat([x, x_0_1], dim=1))
            x_0_3 = self.conv17(x_160_2)
            out = self.out_conv(x_0_3)
            out = self.Tanh(out)

        return out, time_embed


class DSCNet_unetirgan(nn.Module):
    def __init__(
            self,
            n_channels,
            n_classes,
            kernel_size,
            extend_scope,
            if_offset,
            device,
            number,
            dim,
            n_blocks,
            padding_type,
            use_dropout
    ):
        """
        Our DSCNet
        :param n_channels: input channel
        :param n_classes: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        :param device: set on gpu
        :param number: basic layer numbers
        :param dim:
        """
        super().__init__()
        device = device
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.relu = nn.ReLU(inplace=True)
        self.number = number
        """
        The three contributions proposed in our paper are relatively independent. 
        In order to facilitate everyone to use them separately, 
        we first open source the network part of DSCNet. 
        <dim> is a parameter used by multiple templates, 
        which we will open source in the future ...
        """
        self.dim = dim  # This version dim is set to 1 by default, referring to a group of x-axes and y-axes
        """
        Here is our framework. Since the target also has non-tubular structure regions, 
        our designed model also incorporates the standard convolution kernel, 
        for fairness, we also add this operation to compare with other methods (like: Deformable Convolution).
        """
        self.conv00 = EncoderConv(n_channels, self.number)
        self.conv1 = EncoderConvInception(1 * self.number, self.number)
        self.conv20 = EncoderConvInception(self.number, 2 * self.number)
        self.conv3 = EncoderConvInception(2 * self.number, 2 * self.number)

        self.conv40 = EncoderConvInception(2 * self.number, 4 * self.number)
        self.conv5 = EncoderConvInception(4 * self.number, 4 * self.number)

        self.conv60 = EncoderConvInception(4 * self.number, 8 * self.number)
        self.conv7 = EncoderConvInception(8 * self.number, 8 * self.number)

        self.conv120 = EncoderConvInception(12 * self.number, 4 * self.number)
        self.conv13 = EncoderConvInception(4 * self.number, 4 * self.number)

        self.conv140 = DecoderConvInception(6 * self.number, 2 * self.number)
        self.conv15 = DecoderConvInception(2 * self.number, 2 * self.number)

        self.conv160 = DecoderConvInception(3 * self.number, self.number)
        self.conv17 = DecoderConvInception(1 * self.number, self.number)
        self.out_conv = nn.Conv2d(self.number, n_classes, 1)
        ###
        self.sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)
        ###
        ###
        self.z_transform = ZTransform(16 * 16, 8 * self.number, n_out=2)
        self.time_embed = TimeEmbedding(4 * self.number, 8 * self.number)
        self.model_res = nn.ModuleList()
        for i in range(n_blocks):
            self.model_res += [ResnetBlock_cond(8 * self.number, padding_type=padding_type, use_dropout=use_dropout,
                                                temb_dim=8 * self.number, z_dim=8 * self.number)]

    def forward(self, x, time_cond, z, layers=[], encode_only=False):
        ### first embed noise and time information
        z_embed = self.z_transform(z)
        temb = get_timestep_embedding(time_cond, self.number * 4)
        time_embed = self.time_embed(temb)
        if len(layers) > 0:
            feats = []
            # block0
            x_00_0 = self.conv00(x)
            x_0_1 = self.conv1(x_00_0)
            feats.append(x_0_1)

            # block1
            x = Downsample(x_0_1.shape[1])(x_0_1)
            x_20_0 = self.conv20(x)
            x_1_1 = self.conv3(x_20_0)
            feats.append(x_1_1)

            # block2
            x = Downsample(x_1_1.shape[1])(x_1_1)
            x_40_0 = self.conv40(x)
            x_2_1 = self.conv5(x_40_0)
            feats.append(x_2_1)

            # block3
            x = Downsample(x_2_1.shape[1])(x_2_1)
            x_60_0 = self.conv60(x)
            x_3_1 = self.conv7(x_60_0)
            feats.append(x_3_1)

            ### add middle resnet block to encorporate time and z information
            for layer_id, layer in enumerate(self.model_res):
                x_3_1 = layer(x_3_1, time_embed, z_embed)
                if layer_id in layers:
                    feats.append(x_3_1)
                if layer_id == layers[-1] and encode_only:
                    return feats
            return x_3_1, feats

        else:
            x_00_0 = self.conv00(x)
            x_0_1 = self.conv1(x_00_0)
            ##copy feature

            # block1
            x = Downsample(x_0_1.shape[1])(x_0_1)
            x_20_0 = self.conv20(x)
            x_1_1 = self.conv3(x_20_0)

            # block2
            x = Downsample(x_1_1.shape[1])(x_1_1)
            x_40_0 = self.conv40(x)
            x_2_1 = self.conv5(x_40_0)

            # block3
            x = Downsample(x_2_1.shape[1])(x_2_1)
            x_60_0 = self.conv60(x)
            x_3_1 = self.conv7(x_60_0)

            for layer_id, layer in enumerate(self.model_res):
                x_3_1 = layer(x_3_1, time_embed, z_embed)

                # block4
            x = Upsample(x_3_1.shape[1])(x_3_1)
            x_120_2 = self.conv120(torch.cat([x, x_2_1], dim=1))
            x_2_3 = self.conv13(x_120_2)

            # block5
            x = Upsample(x_2_3.shape[1])(x_2_3)
            x_140_2 = self.conv140(torch.cat([x, x_1_1], dim=1))
            x_1_3 = self.conv15(x_140_2)

            # block6
            x = Upsample(x_1_3.shape[1])(x_1_3)
            x_160_2 = self.conv160(torch.cat([x, x_0_1], dim=1))
            x_0_3 = self.conv17(x_160_2)
            out = self.out_conv(x_0_3)
            out = self.Tanh(out)

        return out, time_embed


# class DSCNet_unetddrbgan(nn.Module):
#     def __init__(
#             self,
#             n_channels,
#             n_classes,
#             kernel_size,
#             extend_scope,
#             if_offset,
#             device,
#             number,
#             dim,
#             n_blocks,
#             padding_type,
#             use_dropout
#     ):
#         """
#         Our DSCNet
#         :param n_channels: input channel
#         :param n_classes: output channel
#         :param kernel_size: the size of kernel
#         :param extend_scope: the range to expand (default 1 for this method)
#         :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
#         :param device: set on gpu
#         :param number: basic layer numbers
#         :param dim:
#         """
#         super().__init__()
#         device = device
#         self.kernel_size = kernel_size
#         self.extend_scope = extend_scope
#         self.if_offset = if_offset
#         self.relu = nn.ReLU(inplace=True)
#         self.number = number
#         """
#         The three contributions proposed in our paper are relatively independent.
#         In order to facilitate everyone to use them separately,
#         we first open source the network part of DSCNet.
#         <dim> is a parameter used by multiple templates,
#         which we will open source in the future ...
#         """
#         self.dim = dim  # This version dim is set to 1 by default, referring to a group of x-axes and y-axes
#         """
#         Here is our framework. Since the target also has non-tubular structure regions,
#         our designed model also incorporates the standard convolution kernel,
#         for fairness, we also add this operation to compare with other methods (like: Deformable Convolution).
#         """
#         self.conv00 = EncoderConv(n_channels, self.number)
#         self.conv1 = RRDB(1 * self.number, self.number)
#         self.conv20 = EncoderConv(self.number, 2 * self.number)
#         self.conv3 = RRDB(2 * self.number, 2 * self.number)
#
#         self.conv40 = EncoderConv(2 * self.number, 4 * self.number)
#         self.conv5 = RRDB(4 * self.number, 4 * self.number)
#
#         self.conv60 = EncoderConv(4 * self.number, 8 * self.number)
#         self.conv7 = RRDB(8 * self.number, 8 * self.number)
#
#         self.conv120 = EncoderConv(12 * self.number, 4 * self.number)
#         self.conv13 = RRDB(4 * self.number, 4 * self.number)
#
#         self.conv140 = EncoderConv(6 * self.number, 2 * self.number)
#         self.conv15 = RRDB(2 * self.number, 2 * self.number)
#
#         self.conv160 = EncoderConv(3 * self.number, self.number)
#         self.conv17 = RRDB(1 * self.number, self.number)
#         self.out_conv = nn.Conv2d(self.number, n_classes, 1)
#         ###
#         self.sigmoid = nn.Sigmoid()
#         self.Tanh = nn.Tanh()
#         self.softmax = nn.Softmax(dim=1)
#         self.dropout = nn.Dropout(0.5)
#         ###
#         ###
#         self.z_transform = ZTransform(16 * 16, 8 * self.number, n_out=2)
#         self.time_embed = TimeEmbedding(4 * self.number, 8 * self.number)
#         self.model_res = nn.ModuleList()
#         for i in range(n_blocks):
#             self.model_res += [ResnetBlock_cond(8 * self.number, padding_type=padding_type, use_dropout=use_dropout,
#                                                 temb_dim=8 * self.number, z_dim=8 * self.number)]
#
#     def forward(self, x, time_cond, z, layers=[], encode_only=False):
#         ### first embed noise and time information
#         z_embed = self.z_transform(z)
#         temb = get_timestep_embedding(time_cond, self.number * 4)
#         time_embed = self.time_embed(temb)
#         if len(layers) > 0:
#             feats = []
#             # block0
#             x_00_0 = self.conv00(x)
#             x_0_1 = self.conv1(x_00_0)
#             feats.append(x_0_1)
#
#             # block1
#             x = Downsample(x_0_1.shape[1])(x_0_1)
#             x_20_0 = self.conv20(x)
#             x_1_1 = self.conv3(x_20_0)
#             feats.append(x_1_1)
#
#             # block2
#             x = Downsample(x_1_1.shape[1])(x_1_1)
#             x_40_0 = self.conv40(x)
#             x_2_1 = self.conv5(x_40_0)
#             feats.append(x_2_1)
#
#             # block3
#             x = Downsample(x_2_1.shape[1])(x_2_1)
#             x_60_0 = self.conv60(x)
#             x_3_1 = self.conv7(x_60_0)
#             feats.append(x_3_1)
#
#             ### add middle resnet block to encorporate time and z information
#             for layer_id, layer in enumerate(self.model_res):
#                 x_3_1 = layer(x_3_1, time_embed, z_embed)
#                 if layer_id in layers:
#                     feats.append(x_3_1)
#                 if layer_id == layers[-1] and encode_only:
#                     return feats
#             return x_3_1, feats
#
#         else:
#             x_00_0 = self.conv00(x)
#             x_0_1 = self.conv1(x_00_0)
#             ##copy feature
#
#             # block1
#             x = Downsample(x_0_1.shape[1])(x_0_1)
#             x_20_0 = self.conv20(x)
#             x_1_1 = self.conv3(x_20_0)
#
#             # block2
#             x = Downsample(x_1_1.shape[1])(x_1_1)
#             x_40_0 = self.conv40(x)
#             x_2_1 = self.conv5(x_40_0)
#
#             # block3
#             x = Downsample(x_2_1.shape[1])(x_2_1)
#             x_60_0 = self.conv60(x)
#             x_3_1 = self.conv7(x_60_0)
#
#             for layer_id, layer in enumerate(self.model_res):
#                 x_3_1 = layer(x_3_1, time_embed, z_embed)
#
#                 # block4
#             x = Upsample(x_3_1.shape[1])(x_3_1)
#             x_120_2 = self.conv120(torch.cat([x, x_2_1], dim=1))
#             x_2_3 = self.conv13(x_120_2)
#
#             # block5
#             x = Upsample(x_2_3.shape[1])(x_2_3)
#             x_140_2 = self.conv140(torch.cat([x, x_1_1], dim=1))
#             x_1_3 = self.conv15(x_140_2)
#
#             # block6
#             x = Upsample(x_1_3.shape[1])(x_1_3)
#             x_160_2 = self.conv160(torch.cat([x, x_0_1], dim=1))
#             x_0_3 = self.conv17(x_160_2)
#             out = self.out_conv(x_0_3)
#             out = self.Tanh(out)
#
#         return out, time_embed




class DSCNet_unet_noz(nn.Module):
    def __init__(
            self,
            n_channels,
            n_classes,
            kernel_size,
            extend_scope,
            if_offset,
            device,
            number,
            dim,
            n_blocks,
            padding_type,
            use_dropout
    ):
        """
        Our DSCNet
        :param n_channels: input channel
        :param n_classes: output channel
        :param kernel_size: the size of kernel
        :param extend_scope: the range to expand (default 1 for this method)
        :param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
        :param device: set on gpu
        :param number: basic layer numbers
        :param dim:
        """
        super().__init__()
        device = device
        self.kernel_size = kernel_size
        self.extend_scope = extend_scope
        self.if_offset = if_offset
        self.relu = nn.ReLU(inplace=True)
        self.number = number
        """
        The three contributions proposed in our paper are relatively independent. 
        In order to facilitate everyone to use them separately, 
        we first open source the network part of DSCNet. 
        <dim> is a parameter used by multiple templates, 
        which we will open source in the future ...
        """
        self.dim = dim  # This version dim is set to 1 by default, referring to a group of x-axes and y-axes
        """
        Here is our framework. Since the target also has non-tubular structure regions, 
        our designed model also incorporates the standard convolution kernel, 
        for fairness, we also add this operation to compare with other methods (like: Deformable Convolution).
        """
        self.conv00 = EncoderConv(n_channels, self.number)
        self.conv1 = EncoderConv(1 * self.number, self.number)
        self.conv20 = EncoderConv(self.number, 2 * self.number)
        self.conv3 = EncoderConv(2 * self.number, 2 * self.number)

        self.conv40 = EncoderConv(2 * self.number, 4 * self.number)
        self.conv5 = EncoderConv(4 * self.number, 4 * self.number)

        self.conv60 = EncoderConv(4 * self.number, 8 * self.number)
        self.conv7 = EncoderConv(8 * self.number, 8 * self.number)

        self.conv120 = EncoderConv(12 * self.number, 4 * self.number)
        self.conv13 = EncoderConv(4 * self.number, 4 * self.number)

        self.conv140 = DecoderConv(6 * self.number, 2 * self.number)
        self.conv15 = DecoderConv(2 * self.number, 2 * self.number)

        self.conv160 = DecoderConv(3 * self.number, self.number)
        self.conv17 = DecoderConv(1 * self.number, self.number)
        self.out_conv = nn.Conv2d(self.number, n_classes, 1)
        ###
        self.sigmoid = nn.Sigmoid()
        self.Tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(0.5)
        ###
        ###
        # self.z_transform = ZTransform(16 * 16, 8 * self.number, n_out=2)
        self.time_embed = TimeEmbedding(4 * self.number, 8 * self.number)
        self.model_res = nn.ModuleList()
        for i in range(n_blocks):
            self.model_res += [ResnetBlock_cond_noz(8 * self.number, padding_type=padding_type, use_dropout=use_dropout,
                                                temb_dim=8 * self.number)]

    def forward(self, x, time_cond, layers=[], encode_only=False):
        ### first embed noise and time information
        temb = get_timestep_embedding(time_cond, self.number * 4)
        time_embed = self.time_embed(temb)
        if len(layers) > 0:
            feats = []
            # block0
            x_00_0 = self.conv00(x)
            x_0_1 = self.conv1(x_00_0)
            feats.append(x_0_1)

            # block1
            x = Downsample(x_0_1.shape[1])(x_0_1)
            x_20_0 = self.conv20(x)
            x_1_1 = self.conv3(x_20_0)
            feats.append(x_1_1)

            # block2
            x = Downsample(x_1_1.shape[1])(x_1_1)
            x_40_0 = self.conv40(x)
            x_2_1 = self.conv5(x_40_0)
            feats.append(x_2_1)

            # block3
            x = Downsample(x_2_1.shape[1])(x_2_1)
            x_60_0 = self.conv60(x)
            x_3_1 = self.conv7(x_60_0)
            feats.append(x_3_1)

            ### add middle resnet block to encorporate time and z information
            for layer_id, layer in enumerate(self.model_res):
                x_3_1 = layer(x_3_1, time_embed)
                if layer_id in layers:
                    feats.append(x_3_1)
                if layer_id == layers[-1] and encode_only:
                    return feats
            return x_3_1, feats

        else:
            x_00_0 = self.conv00(x)
            x_0_1 = self.conv1(x_00_0)
            ##copy feature

            # block1
            x = Downsample(x_0_1.shape[1])(x_0_1)
            x_20_0 = self.conv20(x)
            x_1_1 = self.conv3(x_20_0)

            # block2
            x = Downsample(x_1_1.shape[1])(x_1_1)
            x_40_0 = self.conv40(x)
            x_2_1 = self.conv5(x_40_0)

            # block3
            x = Downsample(x_2_1.shape[1])(x_2_1)
            x_60_0 = self.conv60(x)
            x_3_1 = self.conv7(x_60_0)

            for layer_id, layer in enumerate(self.model_res):
                x_3_1 = layer(x_3_1, time_embed)

                # block4
            x = Upsample(x_3_1.shape[1])(x_3_1)
            x_120_2 = self.conv120(torch.cat([x, x_2_1], dim=1))
            x_2_3 = self.conv13(x_120_2)

            # block5
            x = Upsample(x_2_3.shape[1])(x_2_3)
            x_140_2 = self.conv140(torch.cat([x, x_1_1], dim=1))
            x_1_3 = self.conv15(x_140_2)

            # block6
            x = Upsample(x_1_3.shape[1])(x_1_3)
            x_160_2 = self.conv160(torch.cat([x, x_0_1], dim=1))
            x_0_3 = self.conv17(x_160_2)
            out = self.out_conv(x_0_3)
            out = self.Tanh(out)

        return out

class AdaptiveLayer(nn.Module):
    def __init__(self, in_channel, style_dim): 
        super().__init__()

        self.style_net = nn.Linear(style_dim, in_channel * 2) 

        self.style_net.bias.data[:in_channel] = 1
        self.style_net.bias.data[in_channel:] = 0

    def forward(self, input, style): 
        
        style = self.style_net(style).unsqueeze(2).unsqueeze(3) 
        gamma, beta = style.chunk(2, 1)

        out = gamma * input + beta 

        return out 
    
class ZTransform(nn.Module): 
    def __init__(self, input_nc, output_nc, n_out):
        super(ZTransform, self).__init__()
        self.norm = PixelNorm()
        self.layer1 = nn.Linear(input_nc, output_nc)
        self.layers = []
        self.activation = nn.LeakyReLU(0.2)
        for _ in range(n_out):
            self.layers.append(nn.Linear(output_nc, output_nc))
            self.layers.append(nn.LeakyReLU(0.2))
        self.layer2 = nn.Sequential(*self.layers)

    def forward(self, input):
        input = self.norm(input)
        input = self.layer1(input)
        input = self.activation(input)
        output = self.layer2(input)
        return output
    
class TimeEmbedding(nn.Module): 
    def __init__(self, input_nc,output_nc):
        super(TimeEmbedding, self).__init__()
        
        modules_emb = []
        modules_emb.append(nn.Linear(input_nc, output_nc))  
        nn.init.zeros_(modules_emb[-1].bias)
        modules_emb.append(nn.LeakyReLU(0.2))
        modules_emb.append(nn.Linear(output_nc, output_nc))
        nn.init.zeros_(modules_emb[-1].bias)
        modules_emb.append(nn.LeakyReLU(0.2))
        
        self.time_embed = nn.Sequential(*modules_emb)
    
    def forward(self, x):
        return self.time_embed(x)

class ResnetBlock_cond(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, use_dropout,temb_dim,z_dim): 
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock_cond, self).__init__()
        self.conv_block,self.adaptive,self.conv_fin = self.build_conv_block(dim, padding_type, use_dropout,temb_dim,z_dim)
 

    def build_conv_block(self, dim, padding_type, use_dropout,temb_dim,z_dim): 
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer use group normalization
            use_dropout (bool)  -- if use dropout layers.

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        
        self.conv_block = nn.ModuleList()
        self.conv_fin = nn.ModuleList()
        p = 0
        if padding_type == 'reflect':
            self.conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            self.conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        self.conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=False), nn.GroupNorm(dim // 4, dim)]
        ######
        self.adaptive = AdaptiveLayer(dim,z_dim) 
        self.conv_fin += [nn.ReLU(True)]
        if use_dropout:
            self.conv_fin += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            self.conv_fin += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            self.conv_fin += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        self.conv_fin += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=False), nn.GroupNorm(dim // 4, dim)]
        #####
        self.Dense_time = nn.Linear(temb_dim, dim)
        nn.init.zeros_(self.Dense_time.bias)
        #####
        self.style = nn.Linear(z_dim, dim * 2)
        self.style.bias.data[:dim] = 1
        self.style.bias.data[dim:] = 0

        
        return self.conv_block,self.adaptive,self.conv_fin

    def forward(self, x,time_cond,z):

        time_input = self.Dense_time(time_cond) 
        for n,layer in enumerate(self.conv_block):
            out = layer(x)
            if n==0:
                out += time_input[:, :, None, None]
        out = self.adaptive(out,z)
        for layer in self.conv_fin:
            out = layer(out)
        """Forward function (with skip connections)"""
        out = x + out  
        return out


class ResnetBlock_cond_noz(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, use_dropout, temb_dim):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock_cond_noz, self).__init__()
        self.conv_block, self.conv_fin = self.build_conv_block(dim, padding_type, use_dropout, temb_dim,
                                                                             )

    def build_conv_block(self, dim, padding_type, use_dropout, temb_dim):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer use group normalization
            use_dropout (bool)  -- if use dropout layers.

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """

        self.conv_block = nn.ModuleList()
        self.conv_fin = nn.ModuleList()
        p = 0
        if padding_type == 'reflect':
            self.conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            self.conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        self.conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=False), nn.GroupNorm(dim // 4, dim)]
        ######
        # self.adaptive = AdaptiveLayer(dim, z_dim)
        self.conv_fin += [nn.ReLU(True)]
        if use_dropout:
            self.conv_fin += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            self.conv_fin += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            self.conv_fin += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        self.conv_fin += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=False), nn.GroupNorm(dim // 4, dim)]
        #####
        self.Dense_time = nn.Linear(temb_dim, dim)
        nn.init.zeros_(self.Dense_time.bias)
        #####
        # # self.style = nn.Linear(z_dim, dim * 2)
        # self.style.bias.data[:dim] = 1
        # self.style.bias.data[dim:] = 0

        return self.conv_block, self.conv_fin

    def forward(self, x, time_cond):

        time_input = self.Dense_time(time_cond)
        for n, layer in enumerate(self.conv_block):
            out = layer(x)
            if n == 0:
                out += time_input[:, :, None, None]
        # out = self.adaptive(out, z)
        for layer in self.conv_fin:
            out = layer(out)
        """Forward function (with skip connections)"""
        out = x + out
        return out




if __name__ == '__main__':
    a = torch.randn(2, 32, 256, 256)
    b = EncoderConvInception(32, 64)
    c = b(a)
    print(c.shape)
#     #print(out)