import antialiased_cnns
import math
from torchvision import models
import numpy as np
import timm
import torch
from torch import nn
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops import deform_conv2d
from modules.layers import BasicBlock
from utils.generic_utils import upsample
from spherical_conv.submodule import RegularBasicBlock,SphereBasicBlock 


def double_basic_block(num_ch_in, num_ch_out, num_repeats=2):
    layers = nn.Sequential(BasicBlock(num_ch_in, num_ch_out))
    for i in range(num_repeats - 1):
        layers.add_module(f"conv_{i}", BasicBlock(num_ch_out, num_ch_out))
    return layers


class DepthDecoderPP(nn.Module):
    def __init__(
                self, 
                num_ch_enc, 
                scales=range(4), 
                num_output_channels=1,  
                use_skips=True
            ):
        super().__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([64, 64, 128, 256])

        # decoder
        self.convs = nn.ModuleDict()
        # i is encoder depth (top to bottom)
        # j is decoder depth (left to right)
        for j in range(1, 5):
            max_i = 4 - j
            for i in range(max_i, -1, -1):

                num_ch_out = self.num_ch_dec[i]
                total_num_ch_in = 0

                num_ch_in = self.num_ch_enc[i + 1] if j == 1 else self.num_ch_dec[i + 1]
                self.convs[f"diag_conv_{i + 1}{j - 1}"] = BasicBlock(num_ch_in, 
                                                                    num_ch_out)
                total_num_ch_in += num_ch_out

                num_ch_in = self.num_ch_enc[i] if j == 1 else self.num_ch_dec[i]
                self.convs[f"right_conv_{i}{j - 1}"] = BasicBlock(num_ch_in, 
                                                                    num_ch_out)
                total_num_ch_in += num_ch_out

                if i + j != 4:
                    num_ch_in = self.num_ch_dec[i + 1]
                    self.convs[f"up_conv_{i + 1}{j}"] = BasicBlock(num_ch_in, 
                                                                    num_ch_out)
                    total_num_ch_in += num_ch_out

                self.convs[f"in_conv_{i}{j}"] = double_basic_block(
                                                                total_num_ch_in, 
                                                                num_ch_out,
                                                            )

                self.convs[f"output_{i}"] = nn.Sequential(
                BasicBlock(num_ch_out, num_ch_out) if i != 0 else nn.Identity(),
                nn.Conv2d(num_ch_out, self.num_output_channels, 1),
                )

    def forward(self, input_features):
        prev_outputs = input_features
        outputs = []
        depth_outputs = {}
        for j in range(1, 5):
            max_i = 4 - j
            for i in range(max_i, -1, -1):

                inputs = [self.convs[f"right_conv_{i}{j - 1}"](prev_outputs[i])]
                inputs += [upsample(self.convs[f"diag_conv_{i + 1}{j - 1}"](prev_outputs[i + 1]))]

                if i + j != 4:
                    inputs += [upsample(self.convs[f"up_conv_{i + 1}{j}"](outputs[-1]))]

                output = self.convs[f"in_conv_{i}{j}"](torch.cat(inputs, dim=1))
                outputs += [output]

                depth_outputs[f"log_depth_pred_s{i}_b1hw"] = self.convs[f"output_{i}"](output)

            prev_outputs = outputs[::-1]

        return depth_outputs


class CVEncoder(nn.Module):
    def __init__(self, num_ch_cv, num_ch_enc, num_ch_outs):
        super().__init__()

        self.convs = nn.ModuleDict()
        self.num_ch_enc = []

        self.num_blocks = len(num_ch_outs)

        for i in range(self.num_blocks):
            num_ch_in = num_ch_cv if i == 0 else num_ch_outs[i - 1]
            num_ch_out = num_ch_outs[i]
            self.convs[f"ds_conv_{i}"] = BasicBlock(num_ch_in, num_ch_out, 
                                                    stride=1 if i == 0 else 2)

            self.convs[f"conv_{i}"] = nn.Sequential(
                BasicBlock(num_ch_enc[i] + num_ch_out, num_ch_out, stride=1),
                BasicBlock(num_ch_out, num_ch_out, stride=1),
            )
            self.num_ch_enc.append(num_ch_out)

    def forward(self, x, img_feats):
        outputs = []
        for i in range(self.num_blocks):
            x = self.convs[f"ds_conv_{i}"](x)
            x = torch.cat([x, img_feats[i]], dim=1)
            x = self.convs[f"conv_{i}"](x)
            outputs.append(x)
        return outputs

class MLP(nn.Module):
    def __init__(self, channel_list, disable_final_activation = False):
        super(MLP, self).__init__()

        layer_list = []
        for layer_index in list(range(len(channel_list)))[:-1]:
            layer_list.append(
                            nn.Linear(channel_list[layer_index], 
                                channel_list[layer_index+1])
                            )
            layer_list.append(nn.LeakyReLU(inplace=True))

        if disable_final_activation:
            layer_list = layer_list[:-1]

        self.net = nn.Sequential(*layer_list)

    def forward(self, x):
        x = x.float()
        return self.net(x)

class ResnetMatchingEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(
                self, 
                num_layers, 
                num_ch_out, 
                pretrained=True,
                antialiased=True,
            ):
        super().__init__()

        self.num_ch_enc = np.array([64, 64])

        model_source = antialiased_cnns if antialiased else models
        resnets = {18: model_source.resnet18,
                   34: model_source.resnet34,
                   50: model_source.resnet50,
                   101: model_source.resnet101,
                   152: model_source.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers"
                                                            .format(num_layers))

        encoder = resnets[num_layers](pretrained)

        resnet_backbone = [
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool,
            encoder.layer1,
        ]

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.num_ch_out = num_ch_out

        self.net = nn.Sequential(
            *resnet_backbone,
            nn.Conv2d(self.num_ch_enc[-1], 128, (1, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                    128, 
                    self.num_ch_out, 
                    (3, 3), 
                    padding=1, 
                    padding_mode="replicate"
                ),
            nn.InstanceNorm2d(self.num_ch_out)
        )

    def forward(self, input_image):
        return self.net(input_image)

class ResnetMatchingEncoder_ERP(nn.Module):
    """Pytorch module for a resnet encoder with deformable convolutions
    """
    def __init__(
                self, 
                num_layers, 
                num_ch_out, 
                pretrained=True,
                antialiased=True,
            ):
        super().__init__()

        self.num_ch_enc = np.array([64, 64])

        model_source = antialiased_cnns if antialiased else models
        resnets = {18: model_source.resnet18,
                   34: model_source.resnet34,
                   50: model_source.resnet50,
                   101: model_source.resnet101,
                   152: model_source.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers"
                                                            .format(num_layers))

        # Load pretrained ResNet model
        encoder = resnets[num_layers](pretrained)

        #basicBlock = BasicBlock
        resnet_backbone = [
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
            encoder.maxpool,
        ]

        self.first_conv=nn.Sequential(*resnet_backbone)
        self.layer1=encoder.layer1
        self.layer2=SphereBasicBlock(128,256,'ERP',64,64,1,None,1,1)
        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.num_ch_out = num_ch_out
     
        self.net = nn.Sequential(
            nn.Conv2d(64, 128, (1, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(
                    128, 
                    self.num_ch_out, 
                    (3, 3), 
                    padding=1, 
                    padding_mode="replicate"
                ),
            nn.InstanceNorm2d(self.num_ch_out)
        )

        # Replace Conv2d layers in self.net with SphereConv2d, MaxPool2d with SphereMaxPool2d
        #self.replace_conv_with_sphereconv(self.net)

    def forward(self, input_image):
        x=self.first_conv(input_image)
        x1=self.layer1(x)
        x2=self.layer2(x)
        x=x1+x2
        output=self.net(x)
        return output
    

class UNetMatchingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
                                        "mnasnet_100", 
                                        pretrained=True, 
                                        features_only=True,
                                    )

        self.decoder = FeaturePyramidNetwork(
                                        self.encoder.feature_info.channels(), 
                                        out_channels=32,
                                    )
        self.outconv = nn.Sequential(
                                    nn.LeakyReLU(0.2, True),
                                    nn.Conv2d(32, 16, 1),
                                    nn.InstanceNorm2d(16),
                                )

    def forward(self, x):
        encoder_feats = {f"feat_{i}": f for i, f in enumerate(self.encoder(x))}
        return self.outconv(self.decoder(encoder_feats)["feat_1"])



class Encoder(nn.Module):
    """Pytorch module for a image encoder
    """
    def __init__(
                self, 
                #num_ch_out, 
                pretrained=True,
                #antialiased=True,
            ):
        super().__init__()

        self.encoder = timm.create_model(
                                            "tf_efficientnetv2_s_in21ft1k", 
                                            pretrained, 
                                            features_only=True
                                        )
        self.num_ch_enc = self.encoder.feature_info.channels()
        self.feature_hooks = self.encoder.feature_hooks
        self._stage_out_idx=self.encoder._stage_out_idx
        self.conv1=self.encoder.conv_stem
        self.grad_checkpointing = self.encoder.grad_checkpointing
        self.bn1=self.encoder.bn1
        self.blocks = self.encoder.blocks
        self.sphereconv= SphereBasicBlock(512,1024,'ERP',3,24,2,None,1,1)
        #self.conv2=nn.Conv2d(48, 24, (1, 1))  
        
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x2=self.sphereconv(x)
        #x=torch.cat((x1,x2),1)
        x =  x1+x2
        #x=self.conv2(x)
        if self.feature_hooks is None:
            features = []
            if 0 in self._stage_out_idx:
                features.append(x)  # add stem out
            for i, b in enumerate(self.blocks):
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(b, x)
                else:
                    x = b(x)
                if i + 1 in self._stage_out_idx:
                    features.append(x)
            return features
        else:
            self.blocks(x)
            out = self.feature_hooks.get_output(x.device)
            return list(out.values())

        




'''
class DeformConv2dLayer(nn.Module):
    def __init__(self, conv_module):
        super(DeformConv2dLayer, self).__init__()
        # Copy over the convolution parameters
        self.in_channels = conv_module.in_channels
        self.out_channels = conv_module.out_channels
        self.kernel_size = conv_module.kernel_size
        self.stride = conv_module.stride
        self.padding = conv_module.padding
        self.dilation = conv_module.dilation
        self.groups = conv_module.groups
        
        # Convolution weights and bias
        self.weight = nn.Parameter(conv_module.weight.clone())
        self.bias = conv_module.bias is not None
        if self.bias:
            self.bias_param = nn.Parameter(conv_module.bias.clone())
        else:
            self.bias_param = None
        
        self.register_buffer('offset', None)
        

    def forward(self, x):
        # If offset is not initialized, generate it based on input dimensions
        if self.offset is None:
            print("generate offset!!")
            N, C_in, H_in, W_in = x.shape
            self.offset = self.generate_offset(H_in, W_in, x.device, x.dtype)
            self.offset.requires_grad = False
            
        N = x.shape[0]  # Get batch size
        # Expand batch dimension of offset to match input tensor batch size
        offset = self.offset.expand(N, -1, -1, -1)        
        # Apply deformable convolution
        out = deform_conv2d(
            x, offset, self.weight, self.bias_param,
            stride=self.stride, padding=self.padding,
            dilation=self.dilation, mask=None
        )
        return out
    
    def generate_offset(self, H_in, W_in, device, dtype):
        # Calculate output spatial dimensions
        kernel_height, kernel_width = self.kernel_size
        stride_height, stride_width = self.stride
        padding_height, padding_width = self.padding
        dilation_height, dilation_width = self.dilation

        H_out = (
            (H_in + 2 * padding_height - dilation_height * (kernel_height - 1) - 1) // stride_height + 1
        )
        W_out = (
            (W_in + 2 * padding_width - dilation_width * (kernel_width - 1) - 1) // stride_width + 1
        )

        # Generate offsets
        offset = self.distortion_aware_map(
            W_in, H_in, kernel_width, kernel_height,
            stride_width, stride_height, H_out, W_out, device, dtype
        )
        
        return offset

    def distortion_aware_map(self, pano_W, pano_H, k_W, k_H, s_width, s_height, H_out, W_out, device, dtype):
        # Initialize offset tensor
        #offset_channels = 2 * k_H * k_W

        # Create position grid for output feature map
        v_idx = torch.arange(H_out, dtype=dtype, device=device)
        u_idx = torch.arange(W_out, dtype=dtype, device=device)
        v_idx_grid, u_idx_grid = torch.meshgrid(v_idx, u_idx, indexing='ij')

        # Calculate corresponding input image positions
        v = v_idx_grid * s_height
        u = u_idx_grid * s_width

        # Call vectorized equi_coord function
        offsets_x, offsets_y = self.equi_coord(pano_W, pano_H, k_W, k_H, u, v, device, dtype)

        # Combine offsets_x and offsets_y into offset tensor
        offsets = torch.stack([offsets_y, offsets_x], dim=0)  # Shape: [2, k_H * k_W, H_out, W_out]
        print(offsets[:,0,0,0])
        print(offsets[:,0,255,511])
        offsets = offsets.reshape(1, -1, H_out, W_out)        # Shape: [1, 2 * k_H * k_W, H_out, W_out]

        return offsets

    def equi_coord(self, pano_W, pano_H, k_W, k_H, u, v, device, dtype):
        # Shape of u and v: [H_out, W_out]
        H_out, W_out = u.shape

        # Define focal length and center point
        fov_w = k_W * math.radians(360. / pano_W)
        focal = (k_W / 2) / math.tan(fov_w / 2)
        c_x = 0
        c_y = 0

        # Calculate rotation angles phi and theta
        u_r = u - pano_W / 2.
        v_r = v - pano_H / 2.
        phi = u_r / pano_W * 2 * math.pi  # Shape: [H_out, W_out]
        theta = -v_r / pano_H * math.pi   # Shape: [H_out, W_out]

        # Generate local grid
        h_range = torch.arange(k_H, dtype=dtype, device=device) + 0.5 - k_H / 2
        w_range = torch.arange(k_W, dtype=dtype, device=device) + 0.5 - k_W / 2
        h_grid, w_grid = torch.meshgrid(h_range, w_range, indexing='ij')  # Shape: [k_H, k_W]

        # **Use torch.float32 data type when calculating inv_K**
        K = torch.tensor([[focal, 0, c_x],
                        [0, focal, c_y],
                        [0., 0., 1.]], dtype=torch.float32, device=device)
        inv_K = torch.inverse(K)  # Calculate matrix inverse

        # **Convert inv_K back to original data type**
        inv_K = inv_K.to(dtype)

        # Convert local grid to rays
        rays = torch.stack([w_grid, h_grid, torch.ones_like(h_grid)], dim=2)  # Shape: [k_H, k_W, 3]
        rays = rays.reshape(-1, 3).T  # Shape: [3, k_H * k_W]
        rays = inv_K @ rays  # Shape: [3, k_H * k_W]
        rays = rays / torch.norm(rays, dim=0, keepdim=True)  # Normalize

        # Expand rays to match output dimensions
        rays = rays.unsqueeze(0).expand(H_out * W_out, -1, -1)  # Shape: [H_out * W_out, 3, k_H * k_W]

        # Calculate rotation matrix
        sin_phi = torch.sin(phi).reshape(-1)  # Shape: [H_out * W_out]
        cos_phi = torch.cos(phi).reshape(-1)
        sin_theta = torch.sin(theta).reshape(-1)
        cos_theta = torch.cos(theta).reshape(-1)

        zeros = torch.zeros_like(sin_phi)
        ones = torch.ones_like(sin_phi)

        R11 = cos_theta * cos_phi
        R12 = -sin_phi
        R13 = sin_theta * cos_phi
        R21 = cos_theta * sin_phi
        R22 = cos_phi
        R23 = sin_theta * sin_phi
        R31 = -sin_theta
        R32 = zeros
        R33 = cos_theta

        R_row1 = torch.stack([R11, R12, R13], dim=1)  # Shape: [H_out * W_out, 3]
        R_row2 = torch.stack([R21, R22, R23], dim=1)
        R_row3 = torch.stack([R31, R32, R33], dim=1)

        R = torch.stack([R_row1, R_row2, R_row3], dim=1)  # Shape: [H_out * W_out, 3, 3]

        # Rotate rays
        rays_rotated = torch.bmm(R, rays)  # Shape: [H_out * W_out, 3, k_H * k_W]

        # Calculate offsets
        x = pano_W / (2 * math.pi) * torch.atan2(rays_rotated[:, 0, :], rays_rotated[:, 2, :]) + pano_W / 2.
        y = pano_H / math.pi * torch.asin(torch.clamp(rays_rotated[:, 1, :], -1, 1)) + pano_H / 2.

        # Calculate roi_x and roi_y
        roi_x = w_grid.reshape(-1) + u_r.reshape(-1).unsqueeze(-1) + pano_W / 2.
        roi_y = h_grid.reshape(-1) + v_r.reshape(-1).unsqueeze(-1) + pano_H / 2.

        # Calculate offsets
        offsets_x = x - roi_x  # Shape: [H_out * W_out, k_H * k_W]
        offsets_y = y - roi_y  # Shape: [H_out * W_out, k_H * k_W]

        # Reshape to: [k_H * k_W, H_out, W_out]
        offsets_x = offsets_x.T.reshape(k_H * k_W, H_out, W_out)
        offsets_y = offsets_y.T.reshape(k_H * k_W, H_out, W_out)

        return offsets_x, offsets_y
'''    


'''
    def replace_conv_with_deformconv(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                # Replace Conv2d with DeformConv2dLayer
                setattr(module, name, DeformConv2dLayer(child))
            else:
                # Recursively replace submodules
                self.replace_conv_with_deformconv(child)
                
    def replace_conv_with_sphereconv(self, module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
            # Initialize SphereConv2d with original Conv2d parameters
                new_conv = SphereConv2d(
                    in_channels=child.in_channels,
                    out_channels=child.out_channels,
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    groups=child.groups,
                    bias=(child.bias is not None),
                    padding_mode=child.padding_mode
                )
                # Copy weights and bias
                new_conv.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_conv.bias.data.copy_(child.bias.data)
                # Replace module
                setattr(module, name, new_conv)
            elif isinstance(child,nn.MaxPool2d):
            # Assume SphereMaxPool2d initialization is similar
                new_pool = SphereMaxPool2d(
                    kernel_size=child.kernel_size,
                    stride=child.stride,
                    padding=child.padding,
                    dilation=child.dilation,
                    return_indices=child.return_indices,
                    ceil_mode=child.ceil_mode
                )
                setattr(module, name, new_pool)
            else:
                # Recursively replace submodules
                self.replace_conv_with_sphereconv(child)
    '''