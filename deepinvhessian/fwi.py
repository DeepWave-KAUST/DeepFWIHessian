import torch
import numpy as np
import matplotlib.pyplot as plt
import deepwave
from scipy.ndimage import gaussian_filter
from scipy import signal
from tqdm.notebook import tqdm
from typing import Callable, Optional
from utilities import *
from filters import lowpass_filter


class FWIParams:
    """
    A class to hold parameters and functions for Full Waveform Inversion (FWI).

    This class initializes and stores the parameters required for FWI and provides
    methods to calculate source and receiver coordinates, as well as to create wavelets.

    Attributes
    ----------
    nx : int
        Number of grid points in the x-direction.
    nz : int
        Number of grid points in the z-direction.
    dx : float
        Grid spacing in the x-direction.
    nt : int
        Number of time samples.
    dt : float
        Time sampling interval.
    num_dims : int
        Number of dimensions.
    num_shots : int
        Number of shots.
    num_batches : int
        Number of batches.
    num_sources_per_shot : int
        Number of sources per shot.
    num_receivers_per_shot : int
        Number of receivers per shot.
    ds : int or torch.Tensor
        Spacing between sources.
    dr : int or torch.Tensor
        Spacing between receivers.
    sz : torch.Tensor
        Depth of sources.
    rz : torch.Tensor
        Depth of receivers.
    os : float
        Offset of sources.
    orec : float
        Offset of receivers.
    ox : float
        Offset in the x-direction.
    freq : float
        Dominant frequency of the wavelet.
    wavelet : torch.Tensor
        Source wavelet tensor.
    s_cor : torch.Tensor
        Source locations [num_shots, num_sources_per_shot, num_dimensions].
    r_cor : torch.Tensor
        Receiver locations [num_shots, num_receivers_per_shot, num_dimensions].

    Methods
    -------
    get_coordinate(acquisition: int | str) -> tuple[torch.Tensor, torch.Tensor]
        Calculate source and receiver coordinates based on the acquisition mode.
    create_wavelet(wavelet: torch.Tensor, scale: float = 1.0) -> torch.Tensor
        Create a tensor of source amplitudes from the wavelet values with an optional scale.
    """

    def __init__(self, par: dict, wavelet: torch.Tensor, acquisition: int | str):
        """
        Initialize an instance to apply FWI.

        Parameters
        ----------
        par : dict
            Dictionary containing all the parameters of the models used to apply the inversion.
        wavelet : torch.Tensor
            Source wavelet tensor.
        acquisition : int | str
            Type of acquisition. Options are:
            1: Receivers are spread over the whole surface.
            2: Specific offset for receivers.
            'volve': Custom mode for handling Volve dataset.
        """
        # Unpacking and storing parameters
        self.nx = par['nx']
        self.nz = par['nz']
        self.dx = par['dx']
        self.nt = par['nt']
        self.dt = par['dt']
        self.num_dims = par['num_dims']
        self.num_shots = par['num_shots']
        self.num_batches = par['num_batches']
        self.num_sources_per_shot = par['num_sources_per_shot']
        self.num_receivers_per_shot = par['num_receivers_per_shot']
        self.ds = par['ds']
        self.dr = par['dr']
        self.sz = par['sz']
        self.rz = par['rz']
        self.os = par['os']
        self.orec = par['orec']
        self.ot = par['ot']
        self.ox = par['ox']
        self.freq = par['freq']

        self.s_cor, self.r_cor = self.get_coordinate(acquisition)
        self.source_amplitudes = self.create_wavelet(wavelet)



    def get_coordinate(self, mode: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Create arrays containing the source and receiver locations based on the mode.

        Parameters:
        ----------
        mode : int | str
            - 1: Receivers are spread over the whole surface.
            - 2: Specific offset for receivers.
            - 'volve': Custom mode for handling Volve dataset.

        Returns:
        -------
        tuple[torch.Tensor, torch.Tensor]
            Source and Receiver locations:
            - Source locations [num_shots, num_sources_per_shot, num_dimensions].
            - Receiver locations [num_shots, num_receivers_per_shot, num_dimensions].

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
        
        elif mode == 'volve_synthetic':
            x_s = torch.cat([self.sz.reshape(-1, 1), self.ds.reshape(-1, 1)], dim=1).reshape(-1, 1, 2)

            x_r = torch.zeros(self.num_shots, self.num_receivers_per_shot, self.num_dims)
            x_r[0, :, 1] = self.dr
            x_r[:, :, 1] = x_r[0, :, 1].repeat(self.num_shots, 1)
            x_r[0, :, 0] = self.rz
            for idx in range(self.num_shots):
                x_r[idx, :, 0] = self.rz[idx]
            x_s[:,:,1] = x_s[:,:,1]
            x_r[:,:,1] = x_r[:,:,1]
        
        elif mode == 'volve':
            x_s = torch.cat([self.sz.reshape(-1, 1), self.ds.reshape(-1, 1)], dim=1).reshape(-1, 1, 2)

            x_r = torch.zeros(self.num_shots, self.num_receivers_per_shot, self.num_dims)
            x_r[0, :, 1] = self.dr
            x_r[:, :, 1] = x_r[0, :, 1].repeat(self.num_shots, 1)
            x_r[0, :, 0] = self.rz
            for idx in range(self.num_shots):
                x_r[idx, :, 0] = self.rz[idx]
            x_s[:,:,1] = x_s[:,:,1] - 2800
            x_r[:,:,1] = x_r[:,:,1] - 2800
        
        x_s /= self.dx
        x_r /= self.dx

        return x_s, x_r
    
    def create_wavelet(self, wavelet: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
        """
        Creates a scaled wavelet tensor for source amplitudes from the given wavelet values.

        Parameters
        ----------
        wavelet : torch.Tensor
            A tensor of wavelet values that represent the seismic signal to be emitted by the sources.
        scale : float, optional
            A scaling factor applied to the wavelet to adjust its amplitude. Default is 1.0.

        Returns
        -------
        torch.Tensor
            A tensor shaped [num_shots, 1, num_sources_per_shot], representing the scaled source amplitudes.
        """
        source_amplitudes = scale * wavelet.reshape(1, 1, -1).repeat(self.num_shots, 1, self.num_sources_per_shot).float()
        return source_amplitudes


    def create_masks(self, window_size: int = 600, v_direct: float = 1800, more_near_offset_mute: int = None, **kwargs) -> np.ndarray:
        """
        Create masks for seismic data processing with optional near offset muting.

        :param window_size: int, optional
            The size of the window for muting, default is 600 samples.
        :param v_direct: float, optional
            Velocity of the direct wave, default is 1800 m/s.
        :param more_near_offset_mute: int, optional
            Additional muting near the source-receiver offset, specified as the width of the taper to apply.
        :return: np.ndarray
            A 3D array of masks for each shot and receiver with applied muting and tapering.
        """
        masks = np.ones((self.num_shots, self.num_receivers_per_shot, self.nt))
        if torch.is_tensor(self.ds):
            ds = np.round(self.ds[1] - self.ds[0]).item()  # Assuming ds is a torch.Tensor, convert to Python float
            dr = np.round(self.dr[1] - self.dr[0]).item()  # Assuming dr is a torch.Tensor, convert to Python float
        else:
            ds, dr = self.ds, self.dr  # Use directly if not tensors

        ot = kwargs.get('ot', self.ot)  # Get 'ot' from kwargs or use default

        for shot_idx in range(self.num_shots):
            sx = (shot_idx * ds + self.os)
            for receiver_idx in range(self.num_receivers_per_shot):
                rx = (receiver_idx * dr + self.orec)
                dist = abs(sx - rx)
                arrival_time = dist / v_direct / self.dt + ot
                window_start = int(arrival_time) - window_size // 2
                window_end = window_start + window_size

                actual_window_start = max(window_start, 0)
                actual_window_end = min(window_end, self.nt)

                masks[shot_idx, receiver_idx, :actual_window_start] = 0  # Mute before the window

                taper_length = actual_window_end - actual_window_start
                if taper_length > 0:
                    taper = (1 - np.cos(np.linspace(0, np.pi, taper_length))) / 2
                    masks[shot_idx, receiver_idx, actual_window_start:actual_window_end] = taper

                if more_near_offset_mute is not None and abs(shot_idx - receiver_idx) <= 10:
                    taper_t = np.ones(self.nt)
                    width = more_near_offset_mute
                    taper_t[:taper_length+width] = (1 - np.cos(np.linspace(0, np.pi, taper_length+width))) / 2
                    masks[shot_idx, receiver_idx, :] *= taper_t

        return masks

    
def forward_modelling(params: FWIParams, model: torch.Tensor,  device: str):
    """2D acoustic wave equation forward modeling.

    Parameters:
    - model (torch.Tensor): Model tensor.
    - wavelet (torch.Tensor): Wavelet tensor.
    - device (str): Device to perform computation on (e.g., 'cuda', 'cpu').

    Returns:
    - data (torch.Tensor): Seismic data.

    """
    # pml_width parameter control the boundary, for free surface first argument should be 0
    data = deepwave.scalar(model.to(device), params.dx, params.dt,
        source_amplitudes= params.source_amplitudes.to(device),
        source_locations=params.s_cor.to(device),
        receiver_locations=params.r_cor.to(device),
        pml_width=[20, 20, 20, 20],
        accuracy=8,
        # pml_freq=params.freq,
    )[-1]
    return data

def compute_gradient(
    params: FWIParams, 
    model: torch.Tensor, 
    observed_data: torch.Tensor, 
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ignore_n_samples: int, 
    device: str,
    mask: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, float]:
    """
    Compute the gradient of the model parameters using backpropagation, with an option to apply a mask to the data.


    Parameters
    ----------
    params : FWIParams
        Object containing the parameters for computation.
    model : torch.Tensor
        Model tensor.
    observed_data : torch.Tensor
        Observed seismic data tensor.
    loss_function : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Loss function used for computing the loss between predictions and observations.
    ignore_n_samples : int
        Number of time samples to ignore from the beginning of the data.
    device : str
        Device to perform computation on (e.g., 'cuda', 'cpu').
    mask : Optional[torch.Tensor], default=None
        Optional mask tensor to apply to the data. Should be the same shape as the data.

    Returns
    -------
    gradient : torch.Tensor
        Gradient of the model parameters.
    running_loss : float
        Accumulated loss during the computation.
    """

    # Ensure the model and mask (if provided) are on the correct device
    model = model.to(device)
    if mask is not None:
        mask = mask.to(device)
    
    # Initialize running loss
    running_loss = 0.0
    num_batches = params.num_batches
    
    # Ensure gradient is zero before backward pass
    if model.grad is not None:
        model.grad.zero_()
    
    # Loop over shots/batches
    for it in range(num_batches):
        # Simulate the wave propagation using the current model and batch of sources and receivers
        batch_data_pred = deepwave.scalar(
            model,
            params.dx,
            params.dt,
            source_amplitudes=params.source_amplitudes[it::num_batches].to(device),
            source_locations=params.s_cor[it::num_batches].to(device),
            receiver_locations=params.r_cor[it::num_batches].to(device),
            pml_width=[20, 20, 20, 20],
            accuracy=8,
            pml_freq=params.freq
        )[-1]
        
        if mask is not None:
            batch_data_pred *= mask[it::num_batches]
            batch_data_pred = torch.nn.functional.pad(batch_data_pred[:, :, ignore_n_samples:], (0, ignore_n_samples, 0, 0, 0, 0), 'constant', 0)

        # Get the corresponding observed data batch
        batch_observed_data = observed_data[it::num_batches].to(device)

        # Compute loss between predicted and observed data, ignoring the first `ignore_n_samples`
        batch_loss = loss_function(
            batch_data_pred.squeeze(), 
            batch_observed_data.squeeze()
        )
        
        # Backpropagate to compute gradients
        batch_loss.backward()
        
        # Accumulate running loss
        running_loss += batch_loss.item()
        
    # Return a copy of the gradient to avoid modifying it during optimizer step
    return model.grad.clone(), running_loss / num_batches


def compute_dm1(
    params: FWIParams, 
    model: torch.Tensor, 
    dm1: torch.Tensor, 
    loss_function: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ignore_n_samples: int, 
    device: str,
    mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute the gradient of the scattering perturbation (dm1) using Born modeling and backpropagation, with an option to apply a mask to the data.

    Parameters
    ----------
    params : FWIParams
        Object containing the parameters for computation.
    model : torch.Tensor
        Background velocity model tensor.
    dm1 : torch.Tensor
        Scattering perturbation (dm1) tensor.
    loss_function : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        Loss function used for computing the loss between the Born predicted and zero data.
    ignore_n_samples : int
        Number of time samples to ignore from the beginning of the data.
    device : str
        Device to perform computation on (e.g., 'cuda', 'cpu').
    mask : Optional[torch.Tensor], default=None
        Optional mask tensor to apply to the data. Should be the same shape as the data.

    Returns
    -------
    dm1_grad : torch.Tensor
        Gradient of the scattering perturbation (dm1).
    """

    # Ensure model, dm1 and mask (if provided) are on the correct device
    model = model.to(device)
    dm1 = dm1.to(device)
    if mask is not None:
        mask = mask.to(device)

    # Initialize dm1 gradient if not done already
    if dm1.grad is None:
        dm1.grad = torch.zeros_like(dm1)

    num_batches = params.num_batches

    # Zero gradients before the loop
    # dm1.grad.zero_()

    for it in range(num_batches):  # loop over shots
        # Simulate the Born scattered field
        batch_data_pred = deepwave.scalar_born(
            model,
            dm1,
            params.dx,
            params.dt,
            source_amplitudes=params.source_amplitudes[it::num_batches].to(device),
            source_locations=params.s_cor[it::num_batches].to(device),
            receiver_locations=params.r_cor[it::num_batches].to(device),
            pml_width=[20, 20, 20, 20],
            accuracy=8,
            pml_freq=params.freq,
        )[-1].squeeze()
        
        if mask is not None:
            batch_data_pred *= mask[it::num_batches].squeeze()

        # Compute loss against zero data
        batch_loss = loss_function(batch_data_pred[:, ignore_n_samples:], torch.zeros_like(batch_data_pred[:, ignore_n_samples:]))
        # batch_loss = loss_function(batch_data_pred, torch.zeros_like(batch_data_pred))
        
        # Backpropagate to compute gradients
        batch_loss.backward()

    # Return a copy of the gradient to avoid potential in-place modification
    return dm1.grad.clone()


def source_illumination(model: torch.Tensor, source: torch.Tensor, dx: float, dt: float, 
                        x_s: torch.Tensor, device: str) -> torch.Tensor:
    """
    Calculate the source illumination by simulating the wavefield for each shot and summing the energy.

    Parameters:
    - model (torch.Tensor): The velocity model.
    - source (torch.Tensor): Source wavelet.
    - dx (float): Spatial discretization.
    - dt (float): Time sampling interval.
    - x_s (torch.Tensor): Source locations.
    - device (str): Device to perform computation on (e.g., 'cuda', 'cpu').

    Returns:
    - src_illum (torch.Tensor): The source illumination pattern.
    """
    nz, nx = model.shape
    num_shots, _, nt = source.shape
    num_batches = num_shots

    x = torch.arange(nx, dtype=torch.float32) * dx
    z = torch.arange(nz, dtype=torch.float32) * dx
    x, z = torch.meshgrid(x, z, indexing='ij')
    x, z = x.flatten(), z.flatten()
    # 'Receivers' at every point in the model
    x_snap = torch.stack([z, x], dim=-1) / dx
    x_snap = x_snap.unsqueeze(0).expand(num_shots, -1, -1)

    # Move the snapshot grid to the device
    x_snap = x_snap.to(device)

    result = torch.zeros((1, nz*nx, nt), device=device)

    # Simulate the wavefield for each batch and sum up the energy
    for it in tqdm(range(num_shots)):
        source_wavefield = deepwave.scalar(
            model.to(device), 
            dx, 
            dt,
            source_amplitudes=source[it::num_batches].to(device),
            source_locations=x_s[it::num_batches].to(device),
            receiver_locations=x_snap[it::num_batches],
            pml_width=[20, 20, 20, 20],
            accuracy=8,
        )[-1].squeeze()

        # Sum the squared wavefield to result
        result += source_wavefield ** 2
    
    # Sum over time to get the final source illumination
    src_illum = result.sum(dim=-1)
    src_illum = src_illum.squeeze().reshape(nx, nz).T

    return src_illum.to(device)

def process_data(
    params: FWIParams,
    data: np.ndarray, 
    pd: int, 
    fn: float, 
    time_shift: int, 
    window_size: int = 600, 
    v_direct: float = 1800, 
    more_near_offset_mute: Optional[int] = None,
) -> torch.Tensor:
    """
    Process seismic data with padding, tapering, filtering, and masking for Volve data.

    Parameters
    ----------
    params : FWIParams
        Object containing the parameters for computation.
    data: Input seismic data as a NumPy array.
    pd: Padding size.
    fn: Cutoff frequency for the lowpass filter.
    time_shift: Time shift applied to the data.
    window_size: Size of the window for masking, default is 600.
    v_direct: Direct wave velocity for masking, default is 1800 m/s.
    more_near_offset_mute: Additional muting near the source-receiver offset.
    
    :return: Processed data as a torch.Tensor.
    """
    
    # Ensure the time axis is the last one
    if data.shape[0] > data.shape[1]:
        data = np.transpose(data, (1, 2, 0))
    nt = data.shape[2]
    
    # Apply padding and tapering
    data_padded = np.pad(data, ((0, 0), (0, 0), (pd, pd)), mode='edge')
    time_taper = cosine_taper1d(nt + 2 * pd, top_width=pd, bottom_width=pd)
    data_padded *= time_taper
    
    # Apply lowpass filtering
    filtered_data = lowpass_filter(6, fn, data_padded, params.dt, filteringN=2)[..., pd:-pd]
    
    # Generate and apply masks to the direct arrivals
    masks = params.create_masks(window_size=window_size, v_direct=v_direct, more_near_offset_mute=more_near_offset_mute)
    observed_data = filtered_data * masks
    
    # Convert to torch.Tensor and return
    observed_data_tensor = torch.tensor(observed_data).float()
    return observed_data_tensor


def bb_step(deltaX: torch.Tensor, deltaG: torch.Tensor, step_type: str = 'short', epsilon: float = 1e-8) -> float:
    """
    Calculate the step size using the Barzilai-Borwein method with an epsilon for numerical stability.

    The Barzilai-Borwein method provides an estimation for the step size in gradient-based
    optimization algorithms. It attempts to approximate the inverse of the Hessian matrix
    using information from the previous step.

    Parameters:
    - deltaX (torch.Tensor): The difference between the current and previous solutions (x_k - x_{k-1}).
    - deltaG (torch.Tensor): The difference between the current and previous gradients (g_k - g_{k-1}).
    - step_type (str): Determines the type of Barzilai-Borwein step size to compute. It can be either
      'short' for the short-step or 'long' for the long-step. Default is 'short'.
    - epsilon (float): A small value added to the denominator for numerical stability. Default is 1e-8.

    Returns:
    - float: The computed step size as a scalar value.

    Raises:
    - ValueError: If `step_type` is not 'short' or 'long'.

    Example:
    >>> x_k = torch.tensor([1.0, 2.0])
    >>> x_km1 = torch.tensor([0.5, 1.5])
    >>> g_k = torch.tensor([0.1, 0.2])
    >>> g_km1 = torch.tensor([0.1, 0.1])
    >>> bb_step(x_k - x_km1, g_k - g_km1)
    10.0
    """
    if step_type not in ['short', 'long']:
        raise ValueError("step_type must be 'short' or 'long'")

    deltaX, deltaG = deltaX.flatten().float(), deltaG.flatten().float()
    if step_type == 'short':
        numerator = torch.dot(deltaX, deltaG)
        denominator = torch.dot(deltaG, deltaG)
    elif step_type == 'long':
        numerator = torch.dot(deltaX, deltaX)
        denominator = torch.dot(deltaX, deltaG)

    denominator = max(denominator, epsilon)

    alpha = numerator / denominator
    return alpha.item()

def run_fwi(params, model, data, optimizer, loss_fn, freq, FWI_iter, device, *, 
            clip_gradient=None, source_illumination=None, mask=None, taper=None, tsamples=0, bb_step_length=None,
            show_data=False, save_results, exp_name='FWI_exp'):
    model = model.to(device)
    model.requires_grad = True
    gradients, updates, fwi_loss, alphas = [], [], []
    for iteration in tqdm(range(FWI_iter)):
        # Save simulated data
        if iteration == 0 or iteration == FWI_iter - 1:
            data = forward_modelling(params, model.detach().clone(), device).cpu().numpy()
            np.savez(f'{exp_name}/simulated_data_iter_{iteration}_freq_{freq}_grad', data=data)
            if show_data:
                show_3_shots(data, [10, 90, 170], clip=0.02, extent=(params['dr'][0],params['dr'][-1], params['nt']*params['dt'], 0), 
                ylim=(params['nt']*params['dt'], 0), save_path=f'{exp_name}/simulated_data_iter_{iteration}_freq_{freq}.png')
            
        # Compute FWI gradient
        optimizer.zero_grad()
        grad, iter_loss = compute_gradient(params, model, data, loss_fn, tsamples, device)
        fwi_loss.append(iter_loss)
        print(f'FWI iteration: {iteration} loss = {iter_loss}')
        # Clip the gradient values
        if clip_gradient is not None:
            torch.nn.utils.clip_grad_value_(model, torch.quantile(grad.detach().abs(), clip_gradient))
        # Apply source illumination to the gradient
        if source_illumination is not None:
            grad = (grad * model.detach().clone()**3 ) / source_illumination
        if iteration == 0: gmax0 =  torch.abs(grad.detach()).max()
        # Normalize the gradient, mask it around the sources and apply taperinh to the shallower and deeper parts
        grad = (grad /gmax0) * mask * taper
        if mask is not None:
            grad *= mask
        if taper is not None:
            grad *= taper
        gradients.append(grad.cpu().detach().numpy())
        if bb_step_length is not None:
            if iteration > 0:
                delta_model = model.detach().clone() - previous_model
                delta_grad = grad.detach().clone() - previous_grad
                alpha = bb_step(delta_model, delta_grad, 'short')
                optimizer.param_groups[-1]['lr'] = alpha
                alphas.append(alpha)
            # Save the current solution and gradient for calculating the step size in the next iteration
            previous_model = model.detach().clone()
            previous_grad = grad.detach().clone()
        # Update the model
        model.grad.data[:] = grad
        optimizer.step()
        updates.append(model.detach().clone().cpu().numpy())
    # Save the results
    if save_results is not None:
        np.savez(f'{exp_name}/losses_grad', fwi_loss=np.array(fwi_loss),)
        np.savez(f'{exp_name}/gradient', updates=np.array(updates), gradients=np.array(gradients), )
