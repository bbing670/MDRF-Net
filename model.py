import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import calculate_gain

# ==================== MSBlock 模块 ====================

class MSChannel(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super(MSChannel, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class MSConv(nn.Module):
    def __init__(self, in_channels, SK_size=3, strides=1):
        super(MSConv, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=SK_size,
            padding=int(SK_size / 2),
            stride=strides
        )
        self.conv2 = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=SK_size,
            padding=int(SK_size / 2),
            stride=strides
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y)


class MSSpatial(nn.Module):
    def __init__(self, in_channels=64, SK_size=3, kernel_size=3):
        super(MSSpatial, self).__init__()
        self.scnn = MSConv(in_channels=in_channels, SK_size=SK_size)
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.scnn(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MSCat(nn.Module):
    def __init__(self, channel=64):
        super(MSCat, self).__init__()
        self.conv1 = nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(channel)

    def forward(self, x, y):
        z = torch.cat([x, y], dim=1)
        return F.relu(self.bn1(self.conv1(z)))


class MSLayer(nn.Module):
    def __init__(self, channel=64, SK_size=3):
        super(MSLayer, self).__init__()
        self.spatial_ref_calib = MSSpatial(in_channels=channel, SK_size=SK_size)
        self.channel_resp_calib = MSChannel(in_channels=channel)
        self.conv1 = nn.Conv2d(channel * 2, channel, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(channel)

    def forward(self, x_ds, x_ww):
        if x_ww.size(2) != x_ds.size(2) or x_ww.size(3) != x_ds.size(3):
            x_ww = F.interpolate(
                x_ww,
                size=(x_ds.size(2), x_ds.size(3)),
                mode='bilinear',
                align_corners=False
            )
        channel_response = self.channel_resp_calib(x_ds)      # [B, C, 1, 1]
        spatial_reference = self.spatial_ref_calib(x_ww)      # [B, 1, H, W]

        modulation_map = channel_response * spatial_reference # [B, C, H, W]

        z = torch.cat(
            [
                modulation_map * x_ds,
                modulation_map * x_ww
            ],
            dim=1
        )

        return F.relu(self.bn1(self.conv1(z)))


class MSBF(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super(MSBF, self).__init__()
        self.channel = in_channels

        self.msl1 = MSLayer(channel=self.channel, SK_size=3)
        self.msl2 = MSLayer(channel=self.channel, SK_size=5)
        self.msl3 = MSLayer(channel=self.channel, SK_size=7)

        self.ascat1 = MSCat(channel=self.channel)
        self.ascat2 = MSCat(channel=self.channel)

        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.residual_bn = nn.BatchNorm2d(out_channels)

    def forward(self, x_ds, x_ww):
        residual = x_ds

        z3 = self.msl1(x_ds, x_ww)
        z5 = self.msl2(x_ds, x_ww)
        z7 = self.msl3(x_ds, x_ww)

        z35 = self.ascat1(z3, z5)
        z357 = self.ascat2(z35, z7)

        residual = self.residual_bn(self.residual_conv(residual))
        f_msbf = z357 + residual

        return f_msbf

# ==================== DAC 模块 ====================

class DAC(nn.Module):
    def __init__(self, use_cuda=True, in_channels=256, out_channels=256, num_branch=4):
        super(DAC, self).__init__()
        assert in_channels % 4 == 0
        
        self.key_channel = in_channels // 4
        self.padding = 1
        self.kernel_size = 3
        self.in_planes = in_channels
        self.num_branch = num_branch
        
        self.query = nn.Sequential(
            nn.Conv2d(in_channels * 2 + 2, in_channels * 2, kernel_size=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(in_channels * 2, self.key_channel, kernel_size=1, padding=0),
            nn.ReLU(True)
        )
    
        self.memory_key = nn.Conv2d(self.key_channel, self.num_branch, 
                                   kernel_size=1, padding=0, bias=False)
        
        self.memory_value = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) 
             for _ in range(self.num_branch)]
        )
    
        if use_cuda:
            self.memory_key = self.memory_key.cuda()
            self.memory_value = self.memory_value.cuda()
            self.query = self.query.cuda()

    def DAC(self, x, aggregate_weight, aggregate_bias):
        batch_size, in_planes, height, width = x.size()
        x = x.reshape(1, -1, height, width)
    
        output = F.conv2d(
            x, 
            weight=aggregate_weight, 
            bias=aggregate_bias, 
            padding=self.padding, 
            groups=batch_size
        )
        
        output = output.reshape(batch_size, -1, output.size(-2), output.size(-1))
        return output

    def forward(self, x, y, scale, flag):
        batch_size, in_planes, height, width = x.size()
        
        if flag:
            scale_h = torch.ones(1, 1).to(x.device) * scale[0]
            scale_w = torch.ones(1, 1).to(x.device) * scale[1]
            scale_info = torch.cat([scale_h, scale_w], dim=1).repeat(batch_size, 1).unsqueeze(-1).unsqueeze(-1)
        else:
            scale = scale.reshape(batch_size, 1)
            scale_info = scale.repeat(1, 2).unsqueeze(-1).unsqueeze(-1)

        x = x + self.memory_value[0](x)
        y = y + self.memory_value[0](y)

        global_x = F.adaptive_avg_pool2d(x, (1, 1))
        global_y = F.adaptive_avg_pool2d(y, (1, 1))

        global_context = global_x + global_y
        global_difference = global_x - global_y

        query_input = torch.cat([global_context, global_difference, scale_info], dim=1)
        query = self.query(query_input)

        memory_key = self.memory_key.weight
        memory_key = memory_key.view(-1, self.num_branch)
        
        softmax_attention = torch.mm(query[:, :, 0, 0], memory_key)
        softmax_attention = F.softmax(softmax_attention / 20, dim=1).view(x.size(0), -1)

        weight = self.memory_value[0].weight.unsqueeze(0)
        for i in range(1, self.num_branch):
            weight = torch.cat((weight, self.memory_value[i].weight.unsqueeze(0)), dim=0)
        weight = weight.view(self.num_branch, -1)

        aggregate_weight = torch.mm(softmax_attention, weight).view(
            -1, self.in_planes, self.kernel_size, self.kernel_size
        )

        bias = self.memory_value[0].bias.unsqueeze(0)
        for i in range(1, self.num_branch):
            bias = torch.cat((bias, self.memory_value[i].bias.unsqueeze(0)), dim=0)
        bias = bias.view(self.num_branch, -1)
        
        aggregate_bias = torch.mm(softmax_attention, bias).view(-1)

        x = self.DAC(x, aggregate_weight, aggregate_bias) + x
        y = self.DAC(y, aggregate_weight, aggregate_bias) + y
        
        return x, y
    
def DACmodel():
    return DAC(torch.cuda.is_available(), in_channels=64, out_channels=64, num_branch=4)

class SFS(nn.Module):
    def __init__(self, in_planes, ratios=0.25, K=2, temperature=31, init_weight=True):
        super(SFS, self).__init__()

        assert temperature % 3 == 1
        assert K == 2, "For DS/WW state selection, K should be set to 2."

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        if in_planes != 3:
            hidden_planes = int(in_planes * ratios) + 1
        else:
            hidden_planes = K

        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)

        self.register_buffer("temperature", torch.tensor(float(temperature)))

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self):

        with torch.no_grad():
            if self.temperature.item() > 1:
                new_temperature = torch.clamp(self.temperature - 3.0, min=1.0)
                self.temperature.copy_(new_temperature)
    def updata_temperature(self):
        self.update_temperature()

    def get_temperature(self):
        return float(self.temperature.item())

    def forward(self, x_ds, x_ww):
        g_ds = self.avgpool(x_ds)
        g_ww = self.avgpool(x_ww)
        g = g_ds + g_ww
        s = self.fc1(g)
        s = F.relu(s)
        s = self.fc2(s).view(s.size(0), -1)  # [B, 2]

        alpha = F.softmax(s / self.temperature, dim=1)  # [B, 2]

        alpha_ds = alpha[:, 0].view(-1, 1, 1, 1)
        alpha_ww = alpha[:, 1].view(-1, 1, 1, 1)

        f_sfs = alpha_ds * x_ds + alpha_ww * x_ww

        return f_sfs, alpha

# ====================  TCAF 模块 ====================

class TCAF(nn.Module):
    def __init__(self, in_channels, time_steps):
        super(TCAF, self).__init__()
        self.time_steps = time_steps
        self.time_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels // 2, time_steps),
            nn.Softmax(dim=1)
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(time_steps, time_steps, kernel_size=3, padding=1),
            nn.BatchNorm2d(time_steps),
            nn.ReLU(inplace=True),
            nn.Conv2d(time_steps, time_steps, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(time_steps, time_steps // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(time_steps // 4, time_steps, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.feature_enhancement = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )
        
        self.residual_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.residual_bn = nn.BatchNorm2d(in_channels)
        
        self.fusion_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.fusion_bn = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        B, C, T, H, W = x.shape

        x_residual = x[:, :, 0, :, :].clone()
        time_weights = []
        for t in range(T):
            time_feat = x[:, :, t, :, :]              # [B, C, H, W]
            weight = self.time_attention(time_feat)   # [B, T]
            time_weights.append(weight)

        time_attn_matrix = torch.stack(time_weights, dim=1)  # [B, T, T]
        time_weights = time_attn_matrix.mean(dim=1)          # [B, T]
        time_weights = time_weights.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)  # [B, T, H, W]

        spatial_weights = self.spatial_attention(time_weights)  # [B, T, H, W]
        channel_weights = self.channel_attention(time_weights)  # [B, T, 1, 1]
        channel_weights = channel_weights.expand(-1, -1, H, W)  # [B, T, H, W]
        combined_weights = time_weights * spatial_weights * channel_weights  # [B, T, H, W]

        x_weighted = []
        for t in range(T):
            weight = combined_weights[:, t:t+1, :, :]  # [B, 1, H, W]
            feature = x[:, :, t, :, :]                 # [B, C, H, W]
            x_weighted.append(feature * weight)

        x_fused = torch.sum(torch.stack(x_weighted, dim=2), dim=2)  # [B, C, H, W]
        x_enhanced = self.feature_enhancement(x_fused)

        residual1 = self.residual_bn(self.residual_conv(x_residual))
        residual2 = x.mean(dim=2)

        x_fused = x_enhanced + residual1 + residual2
        x_fused = self.fusion_bn(self.fusion_conv(x_fused))
        x_fused = F.relu(x_fused, inplace=True)

        return x_fused
    
# 定义编码器
class Encoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.encoder(x)

# 定义分类器
class Classifier(nn.Module):
    def __init__(self, in_features=64, hidden_features=32, num_classes=2):
        super(Classifier, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden_features, num_classes)
        )
    
    def forward(self, x):
        x = self.global_pool(x).view(x.size(0), -1)
        return self.classifier(x)
# ==================== Fusion Layer ====================

class FusionModule(nn.Module):
    def __init__(self, in_channels=64):
        super(FusionModule, self).__init__()

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.interaction_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 4, in_channels * 2),
            nn.Sigmoid()
        )

    def forward(self, f_msbf, f_sfs):
        h = torch.cat([f_msbf, f_sfs], dim=1)
        h = self.interaction_conv(h)
        h = self.bn(h)
        h = self.relu(h)

        w = self.global_pool(h).view(h.size(0), -1)
        w = self.fc(w).view(h.size(0), -1, 1, 1)

        w1, w2 = torch.split(w, h.size(1), dim=1)

        fused = w1 * f_msbf + w2 * f_sfs
        residual = 0.5 * (f_msbf + f_sfs)
        fused = fused + residual
        return fused


# 定义主模型
class ImagePairClassifier(nn.Module):
    def __init__(self, time_steps=6):
        super(ImagePairClassifier, self).__init__()
        self.time_steps = time_steps

        self.encoder = Encoder(in_channels=3, out_channels=64)
        self.Dycon = DACmodel()
        self.attention = SFS(in_planes=64, ratios=0.25, K=2, temperature=31)

        self.asnet = MSBF(in_channels=64, out_channels=64)
        self.fusion_module = FusionModule(in_channels=64)
        self.classifier = Classifier(in_features=64)

        self.temporal_fusion_exp = TCAF(in_channels=3, time_steps=time_steps)
        self.temporal_fusion_ctrl = TCAF(in_channels=3, time_steps=time_steps)

    def forward(self, img1, img2):
        B, C, T, H, W = img1.shape
        exp_img_fused = self.temporal_fusion_exp(img1)
        ctrl_img_fused = self.temporal_fusion_ctrl(img2)
        exp_feat = self.encoder(exp_img_fused)
        ctrl_feat = self.encoder(ctrl_img_fused)
        scale = torch.ones(B, 1, device=img1.device)
        flag = False
        fused_feat1, fused_feat2 = self.Dycon(exp_feat, ctrl_feat, scale, flag)

        f_sfs, attention_weights = self.attention(fused_feat1, fused_feat2)

        f_msbf = self.asnet(fused_feat1, fused_feat2)

        if f_msbf.size(2) != f_sfs.size(2) or f_msbf.size(3) != f_sfs.size(3):
            f_msbf = F.interpolate(
                f_msbf,
                size=(f_sfs.size(2), f_sfs.size(3)),
                mode='bilinear',
                align_corners=False
            )

        enhanced_feat = self.fusion_module(f_msbf, f_sfs)

        output = self.classifier(enhanced_feat)

        return output