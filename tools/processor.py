import os
import random
import numpy as np
import SimpleITK as sitk
from tools.utils import *


def PreprocessData(patientid, from_dir, to_dir,
                   mask_dir, 
                   crop: bool = True,
                   nor_region: str = "crop"):
    """
    处理所有的图像数据
    :param patientid: 患者id
    :param from_dir: 数据路径
    :param to_dir: 保存路径
    :param mask_dir: TissueMask的路径
    :param selections: 选择的图像
    :param crop: 是否crop
    :param normlization: 是否标准化
    :param nor_region: 标准化区域
    :param explore: 是否保存处理后的图像
    :return:
    """
    # patientid = patientids[1]
    # 判断to_dir是否存在
    if not os.path.exists(os.path.join(to_dir, patientid)):

        save_dir = os.path.join(to_dir, patientid)
        mkdir(save_dir)  # Create a directory to save the processed images

        for np_file in nonHidden_listdir(os.path.join(from_dir, patientid)):

            phases = ["T2DCE_Phase0.nii.gz", 
                      "T2DCE_Phase1.nii.gz", 
                      "T2DCE_Phase4.nii.gz"]
            # 检测所有phase文件是否存在
            all_phases_exist = all(os.path.exists(os.path.join(from_dir, patientid, np_file, phase)) for phase in phases)
            if not all_phases_exist:
                continue

            for phase in phases:
                save_processed_image(patientid, to_dir, crop, nor_region, save_dir, np_file, phase,
                                     phase_img_dir = os.path.join(from_dir, patientid, np_file, phase), 
                                     phase_mask_dir = os.path.join(mask_dir, patientid, np_file, "T2DCE_Phase0.nii.gz"))
    else:
        pass

def save_processed_image(patientid, to_dir, crop, nor_region, save_dir, np_file, phase, phase_img_dir, phase_mask_dir):
    
    if os.path.exists(phase_img_dir) and os.path.exists(phase_mask_dir):
        mkdir(os.path.join(save_dir, np_file))

        phase_maskData = sitk.GetArrayFromImage(sitk.ReadImage(phase_mask_dir))
                
        # 读取Phase0图像
        phase_img = read_mri_image(phase_img_dir)
        # 将Phase0图像转换为数组
        phase_imgData = sitk.GetArrayFromImage(phase_img)
                
        # 对Phase0图像进行处理
        process_phase_imgData = process(phase_imgData, phase_maskData, crop, nor_region)
        process_phase_img = sitk.GetImageFromArray(process_phase_imgData)
        # 保留原始图像的元信息
        process_phase_img.CopyInformation(phase_img)

        # 保存处理后的Phase0影像
        sitk.WriteImage(process_phase_img, os.path.join(to_dir, patientid, np_file, phase))
    else:
        pass



def process(imgData, maskData, crop, nor_region):
    imgData = mri_norm(data=imgData, regionarray=maskData, region=nor_region)
    if crop:  #切不切
        imgData = cropData_to_Seg(imgData, maskData)
    return imgData


def mri_norm(data, regionarray, region):
    """MRI图像标准化"""

    if region == "crop":
        mask_array = regionarray
    elif region == "all":
        mask_array = np.ones(regionarray.shape)

    mask = (mask_array == 1)
    data_cut = data[mask]
    min = data_cut.min()
    max = data_cut.max()

    # 应用Z-Score标准化
    normalized_array = (data - min) / (max - min + 1e-8)

    return normalized_array


def cropData_to_Seg(Data, Seg):
    """
    crop data to seg
    :param data:
    :param Seg:
    :return:
    """

    mask = (Seg == 0)
    Data[mask] = 0

    return Data
