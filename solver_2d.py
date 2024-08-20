import os
import logging
import time
import glob
import json
import sys
sys.path.append("/home/adenker/projects/DDS_CT")

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import torch
import torch.utils.data as data

import torchvision.utils as tvu

from guided_diffusion.models import Model
from guided_diffusion.script_util import create_model, classifier_defaults, args_to_dict
from guided_diffusion.utils import get_alpha_schedule
import random

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from scipy.linalg import orth
from pathlib import Path

from physics.ct import CT
from physics.mri import SinglecoilMRI_comp, MulticoilMRI
from time import time
from utils import shrink, CG, clear, batchfy, get_mask, nchw_comp_to_real, real_to_nchw_comp, normalize_np, get_beta_schedule
from functools import partial



class Diffusion(object):
    def __init__(self, args, config, device=None, coeff_schedule="ddnm"):
        self.args = args
        self.args.image_folder = Path(self.args.image_folder)
        self.coeff_schedule = coeff_schedule
        for t in ["input", "recon", "label"]:
            if t == "recon":
                (self.args.image_folder / t / "progress").mkdir(exist_ok=True, parents=True)
            else:
                (self.args.image_folder / t).mkdir(exist_ok=True, parents=True)
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self, simplified):
        cls_fn = None
        config_dict = vars(self.config.model)
        model = create_model(**config_dict)
        print(self.args.exp)
        ckpt = os.path.join(self.args.exp, "vp", self.args.ckpt_load_name)
        
        model.load_state_dict(torch.load(ckpt, map_location=self.device))
        print(f"Model ckpt loaded from {ckpt}")
        model.convert_to_fp32()
        model.dtype = torch.float32
        model.to("cuda")
        model.eval()

        print('Run Simplified DDNM, without SVD.',
                f'{self.config.time_travel.T_sampling} sampling steps.',
                f'travel_length = {self.config.time_travel.travel_length},',
                f'travel_repeat = {self.config.time_travel.travel_repeat}.',
                f'Task: {self.args.deg}.'
                )
        self.simplified_ddnm_plus(model)
            
            
    def simplified_ddnm_plus(self, model):
        args, config = self.args, self.config
        print(self.args.dataset_path)
        root = Path(self.args.dataset_path)
        
        # parameters to be moved to args
        Nview = self.args.Nview
        rho = self.args.rho
        lamb = self.args.lamb
        n_ADMM = 1
        n_CG = self.args.CG_iter
        
        # read all data
        if "MRI" in args.deg:
            fname_list = sorted(os.listdir(root / "slice"))
            fname_mps_list = sorted(os.listdir(root / "mps"))
        else:
            try:
                fname_list = sorted(fname_list, key=lambda x: float(x.split(".")[0]))
            except ValueError:
                print("files could not be sorted. Continue")
        
        all_img = []

        if self.args.deg == "SV-CT":
            A_funcs = CT(img_width=256, radon_view=self.args.Nview, uniform=True, circle=False, device=config.device)
        elif self.args.deg == "LA-CT":
            A_funcs = CT(img_width=256, radon_view=self.args.Nview, uniform=False, circle=False, device=config.device)
        
        A = lambda z: A_funcs.A(z)
        Ap = lambda z: A_funcs.A_dagger(z)
        
        for idx, fname in enumerate(fname_list):
            if "MRI" in args.deg:
                filename = Path(root) / "slice" / fname
                filename_mps = Path(root) / "mps" / fname
                x_orig = torch.from_numpy(np.load(filename))
            else:
                x_orig = torch.from_numpy(np.load(os.path.join(root, fname), allow_pickle=True))
            
            # For MRI, we have to redefine the forward operator for every slice,
            # as we have different sensitivity maps for multi-coil MRI.
            if args.deg == "MRI-single":
                mask = get_mask(torch.zeros([1, 1, config.data.image_size, config.data.image_size]), config.data.image_size, 
                                config.sampling.batch_size, type=args.mask_type,
                                acc_factor=args.acc_factor, center_fraction=args.center_fraction)
                mask = mask.to(self.device)
                A_funcs = SinglecoilMRI_comp(config.data.image_size, mask)
            elif args.deg == "MRI-multi":
                mask = get_mask(torch.zeros([1, 1, config.data.image_size, config.data.image_size]), config.data.image_size, 
                                config.sampling.batch_size, type=args.mask_type,
                                acc_factor=args.acc_factor, center_fraction=args.center_fraction)
                mask = mask.to(self.device)
                mps = torch.from_numpy(np.load(filename_mps))
                Ncoil, _, _ = mps.shape
                mps = mps.view(1, Ncoil, config.data.image_size, config.data.image_size).to(config.device)
                A_funcs = MulticoilMRI(config.data.image_size, mask, mps)
            A = lambda z: A_funcs._A(z)
            AT = lambda z: A_funcs._AT(z)
            Ap = AT

            h, w = x_orig.shape
            x_orig = x_orig.view(1, 1, h, w)
            x_orig = x_orig.to(self.device)
            y = A(x_orig)
            Apy = Ap(y)
            ATy = AT(y)
            
            Apy_sv = clear(Apy)
            plt.imsave(str(self.args.image_folder / "input" / f"{str(idx).zfill(3)}.png"), np.abs(Apy_sv), cmap='gray')
            x_orig_sv = clear(x_orig)
            plt.imsave(str(self.args.image_folder / "label" / f"{str(idx).zfill(3)}.png"), np.abs(x_orig_sv), cmap='gray')
            
            def Acg_noise(x, gamma):
                return x + gamma * AT(A(x))
            
            def Acg(x):
                return AT(A(x))
            
            if self.args.sigma_y == 0.0:
                Acg_fn = Acg
                bcg = AT(y)
            else:
                Acg_fn = partial(Acg_noise, gamma=self.args.gamma)
                bcg = AT(y)
            
            # init x_T
            x = torch.randn(
                x_orig.shape[0],
                config.data.channels,
                config.data.image_size,
                config.data.image_size,
                device=self.device,
            )
        
            with torch.no_grad():
                skip = config.diffusion.num_diffusion_timesteps//args.T_sampling
                n = x.size(0)
                x0_preds = []
                xs = [x]
                
                # generate time schedule
                times = range(0, 1000, skip)
                times_next = [-1] + list(times[:-1])
                times_pair = zip(reversed(times), reversed(times_next))
                
                # reverse diffusion sampling
                for i, j in tqdm.tqdm(times_pair):
                    t = (torch.ones(n) * i).to(x.device)
                    next_t = (torch.ones(n) * j).to(x.device)
                    
                    at = compute_alpha(self.betas, t.long())
                    at_next = compute_alpha(self.betas, next_t.long())
                    
                    xt = xs[-1].to('cuda')
                    
                    # 0. NFE
                    et = model(xt, t)
                    et = et[:, :et.size(1)//2]

                    # 1. Tweedie
                    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                    
                    if "MRI" in args.deg:
                        x_sv = clear(real_to_nchw_comp(x0_t))
                    else:
                        x_sv = clear(x0_t)
                    plt.imsave(str(self.args.image_folder / "recon" / f"progress" / f"reco_{str(j).zfill(3)}.png"), np.abs(x_sv), cmap='gray')

                    # 2. CG
                    if self.args.sigma_y == 0.0:
                        if "MRI" in args.deg:
                            x0_t = real_to_nchw_comp(x0_t)
                        if args.deg.lower() == "mri-single":
                            x0_t_hat = x0_t - Ap(A(x0_t) - y)
                        else:
                            x0_t_hat = CG(Acg_fn, bcg, x0_t, n_inner=self.args.CG_iter)
                        if "MRI" in args.deg:
                            x0_t_hat = nchw_comp_to_real(x0_t_hat)
                    else:
                        bcg = x0_t + args.gamma * AT(y)
                        if "MRI" in args.deg:
                            x0_t = real_to_nchw_comp(x0_t)
                        if args.deg.lower() == "mri-single":
                            x0_t_hat = x0_t - Ap(A(x0_t) - y)
                        else:
                            x0_t_hat = CG(Acg_fn, bcg, x0_t, n_inner=self.args.CG_iter)
                        if "MRI" in args.deg:
                            x0_t_hat = nchw_comp_to_real(x0_t_hat)

                    eta = self.args.eta

                    if self.coeff_schedule == "ddnm":
                        c1 = (1 - at_next).sqrt() * eta
                        c2 = (1 - at_next).sqrt() * ((1 - eta ** 2) ** 0.5)
                    elif self.coeff_schedule == "ddim":
                        c1 = eta * ((1 - at / at_next) * (1 - at_next)/(1 - at)).sqrt()
                        c2 = (1 - at_next - c1**2).sqrt()

                    # DDIM sampling
                    if j != 0:
                        xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t_hat) + c2 * et
                    # Final step
                    else:
                        xt_next = x0_t_hat

                    x0_preds.append(x0_t.to('cpu'))
                    xs.append(xt_next.to('cpu'))
                x = xs[-1]
            
            if "MRI" in args.deg:
                x_sv = clear(real_to_nchw_comp(x))
            else:
                x_sv = clear(x)
            plt.imsave(str(self.args.image_folder / "recon" / f"{str(idx).zfill(3)}.png"), np.abs(x_sv), cmap='gray')
            # compute PSNR betwee x_orig_sv and x_sv
            psnr = peak_signal_noise_ratio(normalize_np(np.abs(x_orig_sv)),
                                           normalize_np(np.abs(x_sv)))
            summary = {}
            summary["results"] = {"PSNR": psnr}
            with open(str(self.args.image_folder / f"summary.json"), 'w') as f:
                json.dump(summary, f)
            print(f"PSNR: {psnr:.2f}")


def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a
