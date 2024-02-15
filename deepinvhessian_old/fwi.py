import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import deepwave
from scipy.ndimage import gaussian_filter
from scipy import signal


class FWI:
    def __init__(self, par: dict, acquisition: int):
        """Initialize an instance to apply FWI.

        Parameters:
        - par (dict): Containing all the parameters of the models used to apply the inversion.
            - nx (int): Number of grid points in the x-direction.
            - nz (int): Numbe r of grid points in the z-direction.
            - dx (float): Grid spacing in the x-direction.
            - nt (int): Number of time samples.
            - dt (float): Time sampling interval.
            - num_dims (int): Number of dimensions.
            - ns (int): Number of shots.
            - num_batches (int): Number of batches.
            - nr (int): Number of receivers per shot.
            - ds (float): Spacing between sources.
            - dr (float): Spacing between receivers.
            - sz (float): Depth of sources.
            - rz (float): Depth of receivers.
            - osou (float): Offset of sources.
            - orec (float): Offset of receivers.
            - ox (float): Offset in the x-direction.
            - freq (float): Dominant frequency.
        - acquisition (int, optional): Type of acquisition (1 or 2).

        """
        self.nx = par['nx']
        self.nz = par['nz']
        self.dx = par['dx']
        self.nt = par['nt']
        self.dt = par['dt']
        self.num_dims = par['num_dims']
        self.num_shots = par['ns']
        self.num_batches = par['num_batches']
        self.num_sources_per_shot = 1
        self.num_receivers_per_shot = par['nr']
        self.ds = par['ds']
        self.dr = par['dr']
        self.sz = par['sz']
        self.rz = par['rz']
        self.os = par['osou']
        self.orec = par['orec']
        self.ox = par['ox']

        self.s_cor, self.r_cor = self.get_coordinate(acquisition)

    def get_coordinate(self, mode: int):
        """Create arrays containing the source and receiver locations.

        Parameters:
        - mode (int): 1: Receivers are spreaded over the whole surface, 2: Specific offset.

        Returns:
        - torch.Tensor: Source locations [num_shots, num_sources_per_shot, num_dimensions].
        - torch.Tensor: Receiver locations [num_shots, num_receivers_per_shot, num_dimensions].

        """
        x_s = torch.zeros(self.num_shots, self.num_sources_per_shot, self.num_dims)
        x_r = torch.zeros(self.num_shots, self.num_receivers_per_shot, self.num_dims)
        # x direction
        x_s[:, 0, 1] = torch.arange(0, self.num_shots).float() * self.ds + self.os - self.ox
        # z direction
        x_s[:, 0, 0] = self.sz

        if mode == 1:
            # x direction
            x_r[0, :, 1] = torch.arange(0, self.num_receivers_per_shot).float() * self.dr + self.orec
            x_r[:, :, 1] = x_r[0, :, 1].repeat(self.num_shots, 1)
            # z direction
            x_r[:, :, 0] = self.rz

        elif mode == 2:  # fixed spread !!
            # x direction
            x_r[0, :, 1] = torch.arange(self.num_receivers_per_shot).float() * self.dr + self.orec
            x_r[:, :, 1] = x_r[0, :, 1].repeat(self.num_shots, 1) + \
                           torch.arange(0, self.num_shots).repeat(self.num_receivers_per_shot, 1).T * self.ds - self.ox
            # z direction
            x_r[:, :, 0] = self.rz

        return x_s, x_r

    def ricker(self, freq: float):
        """Create a Ricker wavelet.

        Parameters:
        - freq (float): Dominant frequency.

        Returns:
        - np.ndarray: Ricker wavelet.

        """
        return (deepwave.wavelets.ricker(freq, self.nt, self.dt, 1/freq)
                .reshape(-1, 1, 1))

    def forward_modelling(self, model: torch.Tensor, wavelet: torch.Tensor, device: str):
        """2D acoustic wave equation forward modeling.

        Parameters:
        - model (torch.Tensor): Model tensor.
        - wavelet (torch.Tensor): Wavelet tensor.
        - device (str): Device to perform computation on (e.g., 'cuda', 'cpu').

        Returns:
        - data (torch.Tensor): Seismic data.

        """
        # pml_width parameter control the boundary, for free surface first argument should be 0
        prop = self.propagator(model, device)
        data = prop(wavelet.to(device), self.s_cor.to(device), self.r_cor.to(device), self.dt).cpu()
        return data

    def propagator(self, model: torch.Tensor, device: str):
        """Create Deepwave propagator ro run 2D acoustic wave equation forward modeling.

        Parameters:
        - model (torch.Tensor): Model tensor.
        - device (str): Device to perform computation on (e.g., 'cuda', 'cpu').

        Returns:
        - Deepwave propagator.

        """
        return deepwave.scalar.Propagator({'vp': model.to(device)}, 
                                          self.dx, pml_width=[0, 20, 20, 20, 0, 0])

    def run_inversion(self, model: torch.Tensor, model_true: torch.Tensor, data_t: torch.Tensor, 
                      wavelet: torch.Tensor, msk: torch.Tensor, niter: int, 
                      device: str, **kwargs):
        """
        This runs the FWI inversion.

        Arguments:
        - model: torch.Tensor [nz, nx]:
            Initial model for FWI.
        - model_true: torch.Tensor [nz, nx]:
            True model for comparison.
        - data_t: torch.Tensor [nt, ns, nr]:
            Observed seismic data.
        - wavelet: torch.Tensor [nt, 1, 1] or [nt, ns, 1]:
            Wavelet.
        - msk: torch.Tensor [nz, nx]:
            Mask for water layer.
        - niter: int:
            Number of iterations.
        - device: str:
            'gpu' or 'cpu'.

        Returns:
        - np.ndarray: Updated model.
        - np.ndarray: Loss per iteration.
        - np.ndarray: Velocity loss per iteration.
        - np.ndarray: Gradient per iteration.

        """

        model = model.to(device)
        wavelet = wavelet.to(device).contiguous()
        msk = msk.int().to(device)
        model.requires_grad = True

        # Defining objective and step-length
        criterion = torch.nn.MSELoss()
        LR = 1e2
        optimizer = torch.optim.SGD([{'params': [model], 'lr': LR}])

        num_batches = self.num_batches
        num_shots_per_batch = int(self.num_shots / num_batches)
        prop = self.propagator(model, device)
        t_start = time.time()
        loss_iter = []
        vel_loss = []
        updates = []
        gradients = []

        for itr in range(niter):
            running_loss = 0
            optimizer.zero_grad()
            for it in range(num_batches):  # loop over shots
                batch_wavl = wavelet[:, it::num_batches]
                batch_data_t = data_t[:, it::num_batches].to(device)
                batch_x_s = self.s_cor[it::num_batches].to(device)
                batch_x_r = self.r_cor[it::num_batches].to(device)
                batch_data_pred = prop(batch_wavl, batch_x_s, batch_x_r, self.dt)
                loss = criterion(batch_data_pred, batch_data_t)
                if loss.item() == 0.0:
                    updates.append(model.detach().cpu().numpy())
                    return np.array(updates)
                loss.backward()
                running_loss += loss.item()

            if itr == 0:
                gmax0 = (torch.abs(model.grad)).max()  # get max of first itr
            model.grad = model.grad / gmax0 * msk  # normalize

            optimizer.step()
            vel_loss.append(criterion(model, model_true).item())
            loss_iter.append(running_loss)
            print('Iteration:', itr, 'Objective:', running_loss)
            updates.append(model.detach().clone().cpu().numpy())

        # End of FWI iteration
        t_end = time.time()
        print('Runtime in min:', (t_end - t_start) / 60)
        return np.array(updates), np.array(loss_iter), np.array(vel_loss), np.array(gradients)
