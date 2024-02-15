import torch
import numpy as np
import matplotlib.pyplot as plt
import deepwave
import torch.nn as nn
from typing import Tuple

class FWI_LBFGS(nn.Module):
    def __init__(self, data_true: torch.Tensor, source_wavelet: torch.Tensor, x_s: torch.Tensor, 
                 x_r: torch.Tensor, dx: float, dt: float, num_batches: int, 
                 model_dims: Tuple[int, int], msk: torch.Tensor, scaling: float, 
                 device=None, dtype=None):
        """Wrapper for running Full Waveform Inversion (FWI) using Deepwave with the LBFGS optimizer from Scipy.

        Parameters:
        - data_true (torch.Tensor): Observed seismic data tensor.
        - source_wavelet (torch.Tensor): Wavelet tensor.
        - x_s (torch.Tensor): Coordinates of sources.
        - x_r (torch.Tensor): Coordinates of receivers.
        - dx (float): Grid spacing in the x-direction.
        - dt (float): Time sampling interval.
        - num_batches (int): Number of batches.
        - model_dims (tuple[int, int]): Dimensions of the model.
        - msk (torch.Tensor): Mask to apply to the gradient.
        - scaling (float): Scaling factor for the gradient.
        - device (str or torch.device): Device to run the computation on (e.g., 'cuda', 'cpu').
        - dtype (torch.dtype): Data type for the tensors.

        """
        super(FWI_LBFGS, self).__init__()
        self.data_true = data_true.to(device)
        self.source_wavelet = source_wavelet
        self.x_s = x_s
        self.x_r = x_r
        self.dx = dx
        self.dt = dt
        self.num_batches = num_batches
        self.model_dims = model_dims
        self.msk = msk
        self.criterion = torch.nn.MSELoss()
        self.scaling = scaling
        self.running_loss = 0.
        self.nWE = 0 # number of wave equations solved
        self.device = device
        

    def forward(self, model, gradt=False, scipy=False):
        """Perform forward computation for Full Waveform Inversion (FWI) using Deepwave.

        Parameters:
        - model: The model to compute the forward pass on.
        - gradt (bool): Whether to compute the gradient.
        - scipy (bool): Whether to use the Scipy optimizer.

        Returns:
        - If gradt is False:
            - float: The running loss value.
        - If gradt is True and scipy is False:
            - tuple: A tuple containing the running loss value and the gradient.
        - If gradt is True and scipy is True:
            - tuple: A tuple containing the running loss value and the flattened gradient.

        """
        if gradt:
            print('Grad')
        else:
            print('Func')

        # Convert model to torch
        if scipy:
            model = torch.tensor(model.astype(np.float32).reshape(self.model_dims),
                               requires_grad=True, device=self.device)

        # Compute loss/gradient
        running_loss = 0.
        for it in range(self.num_batches):
            data_pred = deepwave.scalar(model,
                                   self.dx,
                                   self.dt,
                                   source_amplitudes=self.source_wavelet[it::self.num_batches].to(self.device),
                                   source_locations=self.x_s[it::self.num_batches].to(self.device),
                                   receiver_locations=self.x_r[it::self.num_batches].to(self.device),
                                   pml_width=[20, 20, 20, 20],
                                   accuracy=8,
                                #    pml_freq=5,
            )[-1]
            batch_data_true = self.data_true[it::self.num_batches].to(self.device)
            self.nWE += 1
            loss = self.criterion(data_pred.squeeze(), batch_data_true.squeeze())
            running_loss += loss.item()
            if gradt:
                loss.backward()
                self.nWE += 1

        if not gradt:
            running_loss = running_loss / self.scaling
            print('Loss', float(running_loss))
            return float(running_loss)

        if gradt:
            # Scale gradient
            running_loss = running_loss / self.scaling
            model.grad = model.grad * self.msk / self.scaling
            gradient = model.grad.detach().cpu().numpy().astype(np.float64)
            print(self.scaling)
            
            if not scipy:
                return running_loss, gradient
            else:
                # plt.figure(figsize=(12, 8))
                # m_min, m_max = np.percentile(gradient, [2, 98])
                # plt.imshow(gradient.reshape(self.model_dims), vmin=m_min, vmax=m_max,
                #            aspect='auto', cmap='bwr')
                # plt.colorbar()
                return running_loss, gradient.ravel()

    def grad(self, model, scipy=False):
        """Compute the gradient for Full Waveform Inversion (FWI) using Deepwave.

        Parameters:
        - model: The model to compute the gradient on.
        - scipy (bool): Whether to use the Scipy optimizer.

        Returns:
        - If scipy is False:
            - tuple: A tuple containing the running loss value and the gradient.
        - If scipy is True:
            - tuple: A tuple containing the running loss value and the flattened gradient.

        """
        return self.forward(model, gradt=True, scipy=scipy)

    def callback(self, x, x_true, mssim, data_residual, nWEs):
        """Callback function for Full Waveform Inversion (FWI) optimization using Scipy.

        Parameters:
        - x (np.ndarray): Current model during optimization.
        - x_true (np.ndarray): True model.
        - velocity_error (list): List to store velocity errors during optimization.
        - data_residual (list): List to store data residuals during optimization.
        - nWEs (list): List to store the number of wave equations solved during optimization.

        Returns:
        - np.ndarray: Velocity errors.
        - np.ndarray: Data residuals.
        - np.ndarray: Number of wave equations solved.

        """
        data_range = x_true.max() - x_true.min()
        data_range = data_range.numpy()
        # x_torch = torch.from_numpy(x)
        # velocity_error.append(self.criterion(x_torch, x_true.reshape(-1)).item())
        x1 = x.reshape(self.model_dims).astype(np.float32)
        x2 =  x_true.numpy()
        # mssim.append(ssim(x1,x2, data_range=data_range))
        xx = torch.from_numpy(x.reshape(self.model_dims)).float().to(self.device)
        loss = self.forward(xx, gradt=False, scipy=False)
        data_residual.append(loss)
        self.nWE = self.nWE - 1
        nWEs.append(self.nWE)
        return np.array(mssim), np.array(data_residual), np.array(nWEs)
