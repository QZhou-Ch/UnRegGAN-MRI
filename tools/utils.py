import os
import shutil
import numpy as np
import pandas as pd
import SimpleITK as sitk
from typing import Tuple, List
from scipy.ndimage import label


def nonHidden_listdir(path):
    """
    Given a specified directory path, return a list of directories under that path.
    Note: Only returns subdirectories under the specified directory.
    
    :param path: The specified directory path.
    :return: A list of subdirectories under the specified directory.
    """
    path_list = os.listdir(path)
    rm_flies = []
    for dir_name in path_list:
        if dir_name.startswith("."):
            rm_flies.append(dir_name)
    for dir_name in rm_flies:
        path_list.remove(dir_name)
    return path_list


def get_dir(plugins_path: str, level: int = 1, del_hiddden: bool = True) -> Tuple[List[str], List[str]]:
    """
    获取指定深度以下的所有文件和目录，返回目录和文件的绝对路径列表

    :param plugins_path: 要遍历的目录
    :type plugins_path: str
    :param level: 遍历的深度，默认为1
    :type level: int
    :param del_hiddden: 是否删除隐藏文件，默认为True
    :type del_hiddden: bool
    :return: 目录和文件列表
    :rtype: Tuple[List[str], List[str]]
    """
    _dirs: List[str] = []
    _files: List[str] = []

    def inner_get_dir(path: str, n: int):
        if n <= 0:
            return
        for df in os.listdir(path):
            abs_path = os.path.join(path, df)
            if os.path.isdir(abs_path):
                _dirs.append(abs_path)
                n -= 1
                inner_get_dir(abs_path, n)
                n += 1
            else:
                _files.append(abs_path)

    def del_files(_list):
        rm_flies = []
        for dir_name in _list:
            if dir_name.split(os.sep)[-1].startswith("."):
                rm_flies.append(dir_name)
        for dir_name in rm_flies:
            _list.remove(dir_name)
        return _list

    inner_get_dir(plugins_path, level)

    if del_hiddden:
        _files = del_files(_files)

    return _dirs, _files


def remkdir(path):
    """
    如果路径存在，则删除路径及其所有子文件夹和文件，然后创建一个新的空文件夹。
    如果路径不存在，则直接创建一个新的空文件夹。

    Args:
        path (str): 要创建的文件夹路径

    Returns:
        None
    """
    if os.path.exists(path):
        shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path)
    else:
        os.makedirs(path)


def mkdir(path):
    """
    如果目录不存在，则创建目录。

    Args:
        path (str): 目录路径。

    Returns:
        None
    """
    if os.path.exists(path):
        pass
    else:
        os.makedirs(path)


def nrrd2nii(nrrd_dir, nii_dir):
    nrrd_files = nonHidden_listdir(nrrd_dir)
    if len(nrrd_files) == 0:
        print("nrrd_dir is empty")
    else:
        for i in nrrd_files:
            _nrrd = sitk.ReadImage(os.path.join(nrrd_dir, i))
            print("saving {}...".format(i.split('.')[0] + '.nii.gz'))
            sitk.WriteImage(_nrrd, os.path.join(nii_dir, (i.split('.')[0] + '.nii.gz')))
            os.remove(os.path.join(nrrd_dir, i))


def read_mri_image(file_path):
    """读取MRI图像并转换为32位浮点类型"""
    image = sitk.ReadImage(file_path)
    return sitk.Cast(image, sitk.sitkFloat32)


def extract_data(file_dir: str, selections: List[str]):
    """
    从给定的文件夹中提取数据，返回一个numpy数组。
    """
    datas = []
    for selection in selections:
        if os.path.exists(os.path.join(file_dir, selection)):
                _dir = os.path.join(file_dir, selection)
                _img = read_mri_image(_dir)
                _data = sitk.GetArrayFromImage(_img)
                datas.append(_data)

    return datas


def segment_relabel(segmented_img_dir, 
                    from_dir, to_dir):
    """
    将分割结果重标记为0,1,2.....
    """
    
    segmented_img = read_mri_image(os.path.join(from_dir, segmented_img_dir))
    patient_imgData = sitk.GetArrayFromImage(segmented_img)

    # 使用label函数对三维数组进行连通组件分析
    # structure定义了体素之间的连通性，这里使用26连通性作为示例
    labeled_array, num_features = label(patient_imgData, structure=np.ones((3,3,3)))

    relabeled_segmented_img = sitk.GetImageFromArray(labeled_array)
    relabeled_segmented_img.CopyInformation(segmented_img)

    relabeled_img_dir = os.path.join(to_dir, ("relabled_" + segmented_img_dir))
    
    sitk.WriteImage(relabeled_segmented_img, relabeled_img_dir)

# 定义一个函数，从文件名中提取关键字符
def extract_key(filename):
    # 检查是不是 'Segmentation' 类型的文件
    if filename.startswith('Segmentation'):
        if filename == 'Segmentation.seg.nrrd':
            return 0
        part = filename.split('_')[-1].split('.')[0]  # 获取 'x' 部分
        return int(part) if part.isdigit() else float('inf')
    # 否则，假设是 'T' 类型的文件
    else:
        order = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'F']
        index = filename.split('.')[0][1:]  # 假设文件名格式是固定的 "Tx.nii.gz"
        return order.index(index) if index in order else float('inf')



def dirs_current(dirs, dropdup=True, n=None, short=False):
    """
    Process directory strings to get current directory paths.
    
    Args:
    dirs (str or list): Directory paths.
    dropdup (bool): Drop duplicate paths.
    n (int): Depth level to process.
    short (bool): Return short or full paths.
    
    Returns:
    list: Processed directory paths.
    """
    if isinstance(dirs, str):
        dir_split = dirs.split(os.sep)
        depth = len(dir_split)
        
        if n is None:
            if short:
                dir_current = dir_split[depth - 1]
            else:
                dir_current = os.sep.join(dir_split[:-1])
        else:
            if short:
                dir_current = dir_split[n:depth]
            else:
                dir_current = os.sep.join(dir_split[:depth - n])
                
    else:
        dir_current = []
        for dir in dirs:
            dir_split = dir.split(os.sep)
            depth = len(dir_split)
            if n is None:
                if short:
                    current = dir_split[depth - 1]
                else:
                    current = os.sep.join(dir_split[:-1])
            else:
                if short:
                    current = dir_split[n:depth]
                else:
                    current = os.sep.join(dir_split[:depth - n])
            dir_current.append(current)
        
        if dropdup:
            dir_current = list(set(dir_current))
    
    return dir_current



# def dirs_current(dirs, dropdup=True, n=None, short=False):
#     dir_split = re.split(r'[\\/]', dirs)
#     depth = len(dir_split)

#     if n is None:
#         if short:
#             dir_current = dir_split[-1]
#         else:
#             dir_current = '/'.join(dir_split[:-1])
#     else:
#         if short:
#             dir_current = '/'.join(dir_split[n:depth])
#         else:
#             dir_current = '/'.join(dir_split[:n])

#     return dir_current
