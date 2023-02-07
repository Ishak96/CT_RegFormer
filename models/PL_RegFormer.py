import numpy as np
import random

import torch
import torch.nn as nn

from pytorch_lightning import LightningModule
from torchvision.utils import save_image

import odl
from odl.contrib import torch as odl_torch

from util.utils import *
from models.LearnFormer import LearnFormer

from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MeanSquaredError

class PL_RegFormer(LightningModule):

    STEPS_CONFIGS = [4, 8]

    def __init__(self, image_size, n_its, lr):
        super(PL_RegFormer, self).__init__()

        self.save_hyperparameters()

        self.automatic_optimization = False

        self.n = image_size
        self.space = odl.uniform_discr([-128, -128], [128, 128], [self.n, self.n], dtype='float32', weighting=1.0)
        
        angle_partition = odl.uniform_partition(0, 2 * np.pi, 512)
        detector_partition = odl.uniform_partition(-360, 360, 512)
        geometry = odl.tomo.geometry.conebeam.FanBeamGeometry(angle_partition, detector_partition, src_radius=256, det_radius=256)
        
        operator = odl.tomo.RayTransform(self.space, geometry)
        self.fp_operator = odl_torch.OperatorModule(operator)
        self.fbp_operator = odl_torch.OperatorModule(operator.adjoint)

        self.learnformer = LearnFormer(self.fp_operator, self.fbp_operator, n_its)

        self.criterion = nn.MSELoss()

    def forward(self, d):
        return self.learnformer(d)

    def LearnFormer_loss(self, y, y_true):
        return self.criterion(y, y_true)

    def step_random_choose(self):
        return random.choice(self.STEPS_CONFIGS)

    def creat_mask(self, proj_data, step):
        theta = np.linspace(0, 512, 512, endpoint=False).astype(int)
        theta_lack = np.linspace(0, 512, int(512/step), endpoint=False).astype(int)
    
        mask_array = np.setdiff1d(theta, theta_lack).astype(int)
    
        mask = np.array(proj_data) != 0
    
        mask[mask_array, :] = 0
    
        return torch.Tensor(mask)    
    
    def fp_ct(self, y_true):
        yy_true = self.space.element(y_true.cpu())
        phantom = torch.Tensor(yy_true)[None, ...]
        
        proj_data = self.fp_operator(phantom)
        
        return proj_data
        
    def get_sparce_sinogram(self, y_true, step):
        proj_data = self.fp_ct(y_true)
        mask = self.creat_mask(torch.squeeze(proj_data), step)
        
        return mask * proj_data

    def training_step(self, batch, batch_idx):
        opt_learnformer = self.optimizers()
        sch_learnformer = self.lr_schedulers()

        y_true = torch.rot90(torch.squeeze(batch), -1)

        step = self.step_random_choose()
        d_sparce = self.get_sparce_sinogram(y_true, step).type_as(y_true)

        y = self(d_sparce)

        learnformer_loss = self.LearnFormer_loss(y, y_true)

        opt_learnformer.zero_grad()
        self.manual_backward(learnformer_loss)
        opt_learnformer.step()

        if self.trainer.is_last_batch:
            sch_learnformer.step()

        self.log_dict({"learnformer_loss": learnformer_loss}, prog_bar=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        y_true = torch.rot90(torch.squeeze(batch), -1)

        step = self.step_random_choose()
        d_sparce = self.get_sparce_sinogram(y_true, step).type_as(y_true)

        y = self(d_sparce)

        learnformer_loss = self.LearnFormer_loss(y, y_true)

        self.log_dict({"val_learnformer_loss": learnformer_loss}, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        psnr, ssim, mse = PeakSignalNoiseRatio().to(device=batch.device), StructuralSimilarityIndexMeasure(data_range=1.0).to(device=batch.device), MeanSquaredError().to(device=batch.device)
        y_true = torch.rot90(torch.squeeze(batch), -1)

        for step in self.STEPS_CONFIGS:
            d_sparce = self.get_sparce_sinogram(y_true, step).type_as(y_true)

            with torch.no_grad():
                self.eval()
                y = self(d_sparce)
                y_true = y_true[None, None, ...]
                self.train()

                test_loss = self.LearnFormer_loss(y, y_true)

                self.log_dict({"test_learnformer_loss": test_loss}, prog_bar=True)

                psnr_p, ssim_p, rmse_p = psnr(y, y_true).cpu().numpy(), ssim(y, y_true).cpu().numpy(), np.sqrt(mse(y, y_true).cpu().numpy())
                file_name = f"results/test/{step}/idx_{batch_idx}_psnr_{psnr_p}_ssim_{ssim_p}_rmse_{rmse_p}.png"

                y = torch.rot90(torch.squeeze(y), 1)
                save_image(y, file_name)
                y_true = torch.squeeze(y_true)

    def configure_optimizers(self):
        lr = self.hparams.lr

        opt_learnformer = torch.optim.AdamW(self.learnformer.parameters(), lr=lr, weight_decay=1e-4)
        sch_learnformer = torch.optim.lr_scheduler.StepLR(opt_learnformer, step_size=150)

        return [opt_learnformer], [sch_learnformer]
