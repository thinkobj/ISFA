import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from networks.aspp import build_aspp
from networks.backbone import build_backbone
from networks.attention import EAM_Module, CAM_Module, PAM_Module


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(24, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.mask_decoder = nn.Sequential(
                                        nn.Conv2d(305, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.5),
                                        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                        BatchNorm(256),
                                        nn.ReLU(),
                                        nn.Dropout(0.1),
                                        nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

        self.edge_decoder = nn.Sequential(nn.Conv2d(48, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                          BatchNorm(256),
                                          nn.ReLU(),
                                          nn.Dropout(0.5),
                                          nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                          BatchNorm(256),
                                          nn.ReLU(),
                                          nn.Dropout(0.1),
                                          nn.Conv2d(256, 1, kernel_size=1, stride=1))
        self.EAM = EAM_Module(in_dim=48)
        self._init_weight()

    def forward(self, x, low_level_feat):

        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        low_level_feat = self.EAM(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        boundary = self.edge_decoder(low_level_feat)

        x = torch.cat([x, boundary], 1)
        x1 = self.mask_decoder(x)

        return x1, boundary

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


##################################################################################
# Basic Blocks
##################################################################################
class ResBlock(nn.Module):

    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):

        super(ResBlock, self).__init__()

        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation=activation, pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1, norm=norm, activation='none', pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):

        residual = x
        out = self.model(x)
        out += residual
        return out


class Conv2dBlock(nn.Module):

    def __init__(self, input_dim ,output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):

        super(Conv2dBlock, self).__init__()

        self.use_bias = True

        # Initialize padding.
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize normalization.
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'ln':
            self.norm = LayerNorm(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize activation.
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize convolution.
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=self.use_bias)

    def forward(self, x):

        x = self.conv(self.pad(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlocks(nn.Module):

    def __init__(self, num_blocks, dim, norm='in', activation='relu', pad_type='zero'):

        super(ResBlocks, self).__init__()

        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim, norm=norm, activation=activation, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):

        return self.model(x)


class StyleEncoder(nn.Module):

    def __init__(self, n_downsample, input_dim, dim, style_dim, norm, activ, pad_type):

        super(StyleEncoder, self).__init__()

        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        for i in range(n_downsample - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)] # Global average pooling.
        self.model += [nn.Conv2d(dim, style_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):

        return self.model(x)


class ContentEncoder(nn.Module):

    def __init__(self, n_downsample, n_res, input_dim, dim, norm, activ, pad_type):

        super(ContentEncoder, self).__init__()

        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3, norm=norm, activation=activ, pad_type=pad_type)]

        # Downsampling blocks.
        for i in range(n_downsample):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        # Residual blocks.
        self.model += [ResBlocks(n_res, dim, norm=norm, activation=activ, pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):

        return self.model(x)


class ISFA(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=2,
                 sync_bn=True, freeze_bn=False):
        super(ISFA, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = Decoder(num_classes, backbone, BatchNorm)
        self.content = ContentEncoder(n_downsample=2, n_res=2, input_dim=24, dim=8, norm='in', activ='relu', pad_type='reflect')
        self.style = StyleEncoder(n_downsample=4, input_dim=24, dim=24, style_dim=8, norm='none', activ='relu', pad_type='reflect')


        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        content_feature = self.content(low_level_feat)
        style_feature = self.style(low_level_feat)

        x = self.aspp(x)

        x1, x2 = self.decoder(x, low_level_feat)

        x2 = F.interpolate(x2, size=input.size()[2:], mode='bilinear', align_corners=True)
        x1 = F.interpolate(x1, size=input.size()[2:], mode='bilinear', align_corners=True)

        return content_feature, style_feature, x1, x2

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p








