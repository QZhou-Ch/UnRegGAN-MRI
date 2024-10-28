from tools.preProcessor import *
from tools.Model import *
from tools.MRIDataset import *
from tools.UnRegGANMRITrainer import *


if __name__ == '__main__':
    root_path = '/home/zhou/NVdisk/MRI_GAN/'
    dali_processor = preProcessor(root_path=root_path)
    dali_processor.FIGURE_PREPROCE()

    print("Done!")

    trainer = UnRegGANMRITrainer(
        input_nc=2,
        output_nc=1,
        lr=2e-4,
        beta1=0.5,
        base_path=root_path + 'DCE_MRI/Crop',
        batch_size=16,
        num_epochs=10,
        log_dir=root_path + 'Results/tensorboard/experiment1',
        save_dir=root_path + 'Results/state_dict'
    )
    trainer.train()
