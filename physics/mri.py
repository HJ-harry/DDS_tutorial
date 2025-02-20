import numpy as np
import torch
import torchkbnufft as tkbn

from utils import fft2, fft2_m, ifft2, ifft2_m


class SinglecoilMRI_real:
    def __init__(self, image_size, mask):
        self.image_size = image_size
        self.mask = mask

    def _A(self, x):
        return fft2(x) * self.mask

    def _Adagger(self, x):
        return torch.real(ifft2(x))

    def _AT(self, x):
        return self._Adagger(x)
    
    
class SinglecoilMRI_comp:
    def __init__(self, image_size, mask):
        self.image_size = image_size
        self.mask = mask

    def _A(self, x):
        return fft2_m(x) * self.mask

    def _Adagger(self, x):
        return ifft2_m(x)

    def _AT(self, x):
        return self._Adagger(x)
    
    
class SinglecoilMRI_NUFFT:
    def __init__(self, image_size, nspokes=30, device='cuda:0'):
        self.image_size = image_size
        self.spokelength = spokelength = image_size * 2
        self.grid_size = (spokelength, spokelength)
        self.nspokes = nspokes
        self.device = device
        
        # create radial trajectory
        ga = np.deg2rad(180 / ((1 + np.sqrt(5)) / 2))
        kx = np.zeros(shape=(spokelength, nspokes))
        ky = np.zeros(shape=(spokelength, nspokes))
        ky[:, 0] = np.linspace(-np.pi, np.pi, spokelength)
        for i in range(1, nspokes):
            kx[:, i] = np.cos(ga) * kx[:, i - 1] - np.sin(ga) * ky[:, i - 1]
            ky[:, i] = np.sin(ga) * kx[:, i - 1] + np.cos(ga) * ky[:, i - 1]
        
        ky = np.transpose(ky)
        kx = np.transpose(kx)

        self.ktraj = np.stack((ky.flatten(), kx.flatten()), axis=0)
        self.ktraj = torch.tensor(self.ktraj).to(self.device)
        print(f"Initialized k-space trajectory with shape: {self.ktraj.shape}")
        
        self.nufft_ob = tkbn.KbNufft(
                im_size=(self.image_size, self.image_size),
                grid_size=self.grid_size,
            ).to(self.device)
        self.adjnufft_ob = tkbn.KbNufftAdjoint(
                im_size=(self.image_size, self.image_size),
                grid_size=self.grid_size,
            ).to(self.device)
        
    def _A(self, x):
        return self.nufft_ob(x, self.ktraj)
    
    def _AT(self, x):
        return self.adjnufft_ob(x, self.ktraj)
    
    def _Adagger(self, x):
        dcomp = tkbn.calc_density_compensation_function(
                    ktraj=self.ktraj, 
                    im_size=(self.image_size, self.image_size)
                )
        return self.adjnufft_ob(x * dcomp, self.ktraj)
        


class MulticoilMRI:
    def __init__(self, image_size, mask, sens):
        self.image_size = image_size
        self.mask = mask
        self.sens = sens

    def _A(self, x):
        return fft2_m(self.sens * x) * self.mask

    def _AT(self, x):
        return torch.sum(torch.conj(self.sens) * ifft2_m(x * self.mask), dim=1).unsqueeze(dim=1)
    
    
def CG(A_fn, b_cg, x, n_inner=10, eps=1e-8):
    r = b_cg - A_fn(x)
    p = r.clone()
    rs_old = torch.matmul(r.view(1, -1), r.view(1, -1).T)
    for _ in range(n_inner):
        Ap = A_fn(p)
        a = rs_old / torch.matmul(p.view(1, -1), Ap.view(1, -1).T)

        x += a * p
        r -= a * Ap

        rs_new = torch.matmul(r.view(1, -1), r.view(1, -1).T)
        
        if torch.sqrt(rs_new) < eps:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x