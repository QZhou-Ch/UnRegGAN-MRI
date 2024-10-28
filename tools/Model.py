import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# 自注意力机制模块（3D版本）
class SelfAttention3D(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention3D, self).__init__()
        self.channel_in = in_dim

        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv   = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim,       kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x):
        '''
        输入:
            x : 输入特征图 (B × C × D × H × W)
        返回:
            out : 自注意力处理后的特征图
        '''
        m_batchsize, C, depth, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, depth * height * width)  # B × C' × N
        proj_key   = self.key_conv(x).view(m_batchsize, -1, depth * height * width)    # B × C' × N
        energy     = torch.bmm(proj_query.permute(0, 2, 1), proj_key)                  # B × N × N
        attention  = self.softmax(energy)                                              # B × N × N
        proj_value = self.value_conv(x).view(m_batchsize, -1, depth * height * width)  # B × C × N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))                        # B × C × N
        out = out.view(m_batchsize, C, depth, height, width)

        out = self.gamma * out + x
        return out

# 残差块（3D版本）
class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock3D, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(channels)
        )

    def forward(self, x):
        residual = x
        out = self.conv_block(x)
        out += residual
        out = F.relu(out)
        return out

# 生成器模型（引入残差连接的3D U-Net）
class UNetGenerator3D(nn.Module):
    def __init__(self, input_nc=2, output_nc=1, ngf=64):
        super(UNetGenerator3D, self).__init__()

        # 编码器部分（增加残差块）
        self.enc1 = self.conv_block(input_nc, ngf, kernel_size=4, stride=2, padding=1)      # [32,32,32] -> [16,16,16]
        self.res1 = ResidualBlock3D(ngf)

        self.enc2 = self.conv_block(ngf, ngf * 2, kernel_size=4, stride=2, padding=1)       # [16,16,16] -> [8,8,8]
        self.res2 = ResidualBlock3D(ngf * 2)

        self.enc3 = self.conv_block(ngf * 2, ngf * 4, kernel_size=4, stride=2, padding=1)   # [8,8,8] -> [4,4,4]
        self.res3 = ResidualBlock3D(ngf * 4)

        # 自注意力层
        self.attn = SelfAttention3D(ngf * 4)

        # 解码器部分（增加残差块）
        self.dec3 = self.up_conv_block(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1)  # [4,4,4] -> [8,8,8]
        self.res_dec3 = ResidualBlock3D(ngf * 2)
        
        self.dec2 = self.up_conv_block(ngf * 4, ngf, kernel_size=4, stride=2, padding=1)      # [8,8,8] -> [16,16,16]
        self.res_dec2 = ResidualBlock3D(ngf)
        
        self.dec1 = nn.Sequential(
            nn.ConvTranspose3d(ngf * 2, output_nc, kernel_size=4, stride=2, padding=1),       # [16,16,16] -> [32,32,32]
            nn.Tanh()
        )

    def conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return block

    def up_conv_block(self, in_channels, out_channels, kernel_size, stride, padding):
        block = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        # 编码器
        e1 = self.enc1(x)   # e1.shape -> [B, ngf, 16, 16, 16]
        e1 = self.res1(e1)

        e2 = self.enc2(e1)  # e2.shape -> [B, ngf*2, 8, 8, 8]
        e2 = self.res2(e2)

        e3 = self.enc3(e2)  # e3.shape -> [B, ngf*4, 4, 4, 4]
        e3 = self.res3(e3)

        # 自注意力
        attn = self.attn(e3)

        # 解码器
        d3 = self.dec3(attn)           # d3.shape -> [B, ngf*2, 8, 8, 8]
        d3 = self.res_dec3(d3)
        d3 = torch.cat([d3, e2], dim=1)  # [B, ngf*4, 8, 8, 8]

        d2 = self.dec2(d3)             # d2.shape -> [B, ngf, 16, 16, 16]
        d2 = self.res_dec2(d2)
        d2 = torch.cat([d2, e1], dim=1)  # [B, ngf*2, 16, 16, 16]

        out = self.dec1(d2)            # out.shape -> [B, output_nc, 32, 32, 32]

        return out

# 判别器模型（与之前相同）
class PatchGANDiscriminator3D(nn.Module):
    def __init__(self, input_nc=1, ndf=64):
        super(PatchGANDiscriminator3D, self).__init__()
        self.model = nn.Sequential(
            # 输入层
            nn.Conv3d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # 隐藏层
            nn.Conv3d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 输出层
            nn.Conv3d(ndf * 4, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# 感知损失（Perceptual Loss，针对3D数据）
class PerceptualLoss3D(nn.Module):
    def __init__(self):
        super(PerceptualLoss3D, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = nn.Sequential(*list(vgg16.features)[:16]).eval()
        for param in self.features.parameters():
            param.requires_grad = False

        # VGG网络的均值和标准差
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, generated, target):
        # 对每个切片计算感知损失
        loss = 0
        for i in range(generated.size(2)):
            gen_slice = generated[:, :, i, :, :]  # [B, C, H, W]
            tgt_slice = target[:, :, i, :, :]     # [B, C, H, W]

            # 将单通道转换为三通道
            gen_slice = gen_slice.repeat(1, 3, 1, 1)
            tgt_slice = tgt_slice.repeat(1, 3, 1, 1)

            # 将数据归一化到[0,1]
            gen_slice = (gen_slice - gen_slice.min()) / (gen_slice.max() - gen_slice.min() + 1e-8)
            tgt_slice = (tgt_slice - tgt_slice.min()) / (tgt_slice.max() - tgt_slice.min() + 1e-8)

            # 使用VGG的均值和标准差进行标准化
            gen_slice = (gen_slice - self.mean) / self.std
            tgt_slice = (tgt_slice - self.mean) / self.std

            # 计算特征
            gen_features = self.features(gen_slice)
            tgt_features = self.features(tgt_slice)

            # 计算L1损失
            loss += F.l1_loss(gen_features, tgt_features)
        loss = loss / generated.size(2)
        return loss

# 计算 PSNR
def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return 100
    max_pixel = 1.0  # 图像已归一化到 [0,1]
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

# 损失函数
def generator_loss(disc_fake_output, gen_output, target, perceptual_loss_fn, lambda_adv=1.0, lambda_l1=100.0, lambda_perc=10.0):
    # 对抗损失
    adv_loss = F.binary_cross_entropy(disc_fake_output, torch.ones_like(disc_fake_output))
    # 重构损失（L1损失）
    l1_loss = F.l1_loss(gen_output, target)
    # 感知损失
    perc_loss = perceptual_loss_fn(gen_output, target)
    # 总损失
    loss = lambda_adv * adv_loss + lambda_l1 * l1_loss + lambda_perc * perc_loss
    return loss, adv_loss.item(), l1_loss.item(), perc_loss.item()

def discriminator_loss(disc_real_output, disc_fake_output):
    real_loss = F.binary_cross_entropy(disc_real_output, torch.ones_like(disc_real_output))
    fake_loss = F.binary_cross_entropy(disc_fake_output, torch.zeros_like(disc_fake_output))
    total_loss = (real_loss + fake_loss) * 0.5
    return total_loss

# 权重初始化
def weights_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
