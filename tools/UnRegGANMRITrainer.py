from tools.Model import *
from tools.MRIDataset import *
import torch
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

class UnRegGANMRITrainer:
    def __init__(self, input_nc, output_nc, lr, beta1, base_path, 
                 batch_size=8, 
                 num_epochs=100, 
                 log_dir='runs/experiment1', save_dir='checkpoints'):
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.lr = lr
        self.beta1 = beta1
        self.base_path = base_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0
        self.save_dir = save_dir

        # Create directory for saving models if it doesn't exist
        os.makedirs(self.save_dir, exist_ok=True)

        # Initialize models
        self.generator = UNetGenerator3D(input_nc=input_nc, output_nc=output_nc).to(self.device)
        self.discriminator = PatchGANDiscriminator3D(input_nc=output_nc, ndf=64).to(self.device)
        self.perceptual_loss_fn = PerceptualLoss3D().to(self.device)

        # Initialize weights
        self.generator.apply(weights_init)
        self.discriminator.apply(weights_init)

        # Initialize optimizers
        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        self.disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

        # Load data
        self.data_loader = self._create_data_loader()
        
        # Print all parameters
        print(f"Input channels: {self.input_nc}")
        print(f"Output channels: {self.output_nc}")
        print(f"Learning rate: {self.lr}")
        print(f"Beta1: {self.beta1}")
        print(f"Base path: {self.base_path}")
        print(f"Batch size: {self.batch_size}")
        print(f"Number of epochs: {self.num_epochs}")
        print(f"Device: {self.device}")
        print(f"Log directory: {log_dir}")
        print(f"Save directory: {self.save_dir}")

    def _get_image_paths(self, phase_name):
        paths = []
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file == phase_name:
                    paths.append(os.path.join(root, file))
        return paths

    def _create_data_loader(self):
        arterial_paths = self._get_image_paths('T2DCE_Phase1.nii.gz')
        delayed_paths = self._get_image_paths('T2DCE_Phase4.nii.gz')
        non_contrast_paths = self._get_image_paths('T2DCE_Phase0.nii.gz')

        assert len(arterial_paths) == len(delayed_paths) == len(non_contrast_paths), "数据路径列表长度不一致！"

        dataset = MRIDataset3D(arterial_paths, delayed_paths, non_contrast_paths, transform=None)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=16)

    def train(self):
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}:")
            self.generator.train()
            self.discriminator.train()
            total_psnr = 0
            batch_count = 0

            pbar = tqdm(enumerate(self.data_loader), total=len(self.data_loader), unit="batch")
            for i, data in pbar:
                arterial_phase = data['arterial'].to(self.device)
                delayed_phase = data['delayed'].to(self.device)
                real_images = data['non_contrast'].to(self.device)
                inputs = torch.cat([arterial_phase, delayed_phase], dim=1)

                ### Update Discriminator ###
                self.disc_optimizer.zero_grad()
                with torch.no_grad():
                    fake_images = self.generator(inputs)
                # Real images
                disc_real_output = self.discriminator(real_images)
                # Fake images
                disc_fake_output = self.discriminator(fake_images.detach())
                # Compute loss
                disc_loss = discriminator_loss(disc_real_output, disc_fake_output)
                disc_loss.backward()
                self.disc_optimizer.step()

                ### Update Generator ###
                self.gen_optimizer.zero_grad()
                fake_images = self.generator(inputs)
                disc_fake_output_for_G = self.discriminator(fake_images)
                gen_loss, adv_loss_val, l1_loss_val, perc_loss_val = generator_loss(
                    disc_fake_output_for_G, fake_images, real_images, self.perceptual_loss_fn)
                gen_loss.backward()
                self.gen_optimizer.step()

                with torch.no_grad():
                    fake_images_clipped = torch.clamp(fake_images, 0, 1)
                    real_images_clipped = torch.clamp(real_images, 0, 1)
                    psnr = calculate_psnr(fake_images_clipped, real_images_clipped)
                    total_psnr += psnr
                    batch_count += 1

                pbar.set_description(f"Batch {i+1}/{len(self.data_loader)}")
                pbar.set_postfix({
                    'D Loss': f"{disc_loss.item():.4f}",
                    'G Loss': f"{gen_loss.item():.4f}",
                    'PSNR': f"{psnr:.2f} dB"
                })

                self.writer.add_scalar('Loss/Discriminator', disc_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/Generator', gen_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/Generator_Adv', adv_loss_val, self.global_step)
                self.writer.add_scalar('Loss/Generator_L1', l1_loss_val, self.global_step)
                self.writer.add_scalar('Loss/Generator_Perceptual', perc_loss_val, self.global_step)
                self.writer.add_scalar('Accuracy/PSNR', psnr, self.global_step)

                if self.global_step % 10 == 0:
                    mid_slice = fake_images.size(2) // 2
                    fake_image_slice = fake_images[0, 0, mid_slice, :, :].detach().cpu().unsqueeze(0)
                    real_image_slice = real_images[0, 0, mid_slice, :, :].detach().cpu().unsqueeze(0)
                    fake_image_slice = (fake_image_slice - fake_image_slice.min()) / (fake_image_slice.max() - fake_image_slice.min() + 1e-8)
                    real_image_slice = (real_image_slice - real_image_slice.min()) / (real_image_slice.max() - real_image_slice.min() + 1e-8)
                    self.writer.add_image('Generated Image', fake_image_slice, self.global_step)
                    self.writer.add_image('Real Image', real_image_slice, self.global_step)

                self.global_step += 1

            avg_psnr = total_psnr / batch_count if batch_count > 0 else 0
            print(f"Epoch [{epoch+1}/{self.num_epochs}] Average PSNR: {avg_psnr:.2f} dB")
            self.writer.add_scalar('Accuracy/Average PSNR per Epoch', avg_psnr, epoch+1)
            torch.save(self.generator.state_dict(), os.path.join(self.save_dir, f'generator_epoch_{epoch+1}.pth'))
            torch.save(self.discriminator.state_dict(), os.path.join(self.save_dir, f'discriminator_epoch_{epoch+1}.pth'))

        torch.save(self.generator.state_dict(), os.path.join(self.save_dir, 'generator_final.pth'))
        torch.save(self.discriminator.state_dict(), os.path.join(self.save_dir, 'discriminator_final.pth'))
        self.writer.close()
