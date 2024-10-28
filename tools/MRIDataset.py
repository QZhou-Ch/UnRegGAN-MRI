import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import torch.nn.functional as F

# 自定义数据集
class MRIDataset3D(Dataset):
    def __init__(self, arterial_paths, delayed_paths, non_contrast_paths, transform=None):
        self.arterial_paths = arterial_paths
        self.delayed_paths = delayed_paths
        self.non_contrast_paths = non_contrast_paths
        self.transform = transform

    def __len__(self):
        # 返回数据集的大小
        return len(self.arterial_paths)
    
    def crop_or_pad(self, image, target_size):
        # image 是一个 PyTorch 张量，形状为 [C, D, H, W]
        _, D, H, W = image.size()
        target_D, target_H, target_W = target_size

        # 计算需要填充或裁剪的大小
        pad_D = max(target_D - D, 0)
        pad_H = max(target_H - H, 0)
        pad_W = max(target_W - W, 0)

        # 如果需要填充，进行对称填充
        if pad_D > 0 or pad_H > 0 or pad_W > 0:
            padding = (pad_W // 2, pad_W - pad_W // 2,  # 左右
                       pad_H // 2, pad_H - pad_H // 2,  # 上下
                       pad_D // 2, pad_D - pad_D // 2)  # 前后
            # F.pad 的填充顺序为 (W 前, W 后, H 前, H 后, D 前, D 后)
            image = F.pad(image, padding, mode='constant', value=0)
            _, D, H, W = image.size()

        # 如果需要裁剪，进行随机裁剪
        if D > target_D:
            start_D = np.random.randint(0, D - target_D + 1)
        else:
            start_D = 0
        if H > target_H:
            start_H = np.random.randint(0, H - target_H + 1)
        else:
            start_H = 0
        if W > target_W:
            start_W = np.random.randint(0, W - target_W + 1)
        else:
            start_W = 0

        cropped_image = image[:, start_D:start_D+target_D, start_H:start_H+target_H, start_W:start_W+target_W]

        return cropped_image

    def __getitem__(self, idx):
        # 读取 NIfTI 格式的 3D MRI 影像
        arterial_image = nib.load(self.arterial_paths[idx]).get_fdata()
        delayed_image = nib.load(self.delayed_paths[idx]).get_fdata()
        non_contrast_image = nib.load(self.non_contrast_paths[idx]).get_fdata()

        # 转换为 Tensor，并添加通道维度
        arterial_image = torch.from_numpy(arterial_image).float().unsqueeze(0)
        delayed_image = torch.from_numpy(delayed_image).float().unsqueeze(0)
        non_contrast_image = torch.from_numpy(non_contrast_image).float().unsqueeze(0)

        # 定义目标尺寸
        target_size = (64, 64, 64)  # 可以根据实际情况调整

        # 对图像进行裁剪或填充
        arterial_image = self.crop_or_pad(arterial_image, target_size)
        delayed_image = self.crop_or_pad(delayed_image, target_size)
        non_contrast_image = self.crop_or_pad(non_contrast_image, target_size)

        # 可选的预处理操作，例如数据增强
        if self.transform:
            arterial_image = self.transform(arterial_image)
            delayed_image = self.transform(delayed_image)
            non_contrast_image = self.transform(non_contrast_image)

        sample = {
            'arterial': arterial_image,
            'delayed': delayed_image,
            'non_contrast': non_contrast_image
        }
        return sample
