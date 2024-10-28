import sys
import pickle
import pandas as pd
from tqdm import tqdm
import multiprocessing
from tools.utils import *
from tools.ptqdm import ptqdm
from multiprocessing import Pool
from sklearn import model_selection as sk_model_selection
from tools.processor import *

class preProcessor(object):
    """
    数据处理类
    """

    def __init__(self, root_path: str,
                 nor_region: str = "crop",
                 crop: bool = True,
                 pall: bool = True):

        self.root_path = root_path
        self.nii_path = os.path.join(root_path, "Nii")
        self.patientids = nonHidden_listdir(self.nii_path)  # 给出指定目录下文件列表
        assert len(self.patientids) != 0, "No data found,can not DALI"

        self.crop = crop
        self.nor_region = nor_region
        self.mask_dir = os.path.join(self.root_path, "Tissue_Mask")
        self.pall = pall
        # 指定多核心数量
        if self.pall:
            if len(self.patientids) > multiprocessing.cpu_count():
                self.n_jobs = int(multiprocessing.cpu_count() - 2)
            else:
                self.n_jobs = len(self.patientids)

        # 存放执行全流程的 DALI 法后的所有文件
        self.DALI_dir = os.path.join(self.root_path, "DCE_MRI")
        mkdir(self.DALI_dir)

        # 存放执行全流程的 DALI 法后的figure/Nii文件
        if self.crop:
            self.tag = "Crop"
        else:
            self.tag = "noCrop"
        print(f'Project tag is " {self.tag} " ')

        self.figure_dir = os.path.join(self.DALI_dir, self.tag)
        mkdir(self.figure_dir)

    def FIGURE_PREPROCE(self):
            ptqdm(PreprocessData, self.patientids,
                  desc="Preprocessing data",
                  from_dir=self.nii_path, to_dir=self.figure_dir,
                  mask_dir=self.mask_dir, 
                  processes=self.n_jobs)