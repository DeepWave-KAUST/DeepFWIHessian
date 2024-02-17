import gc
import numpy as np
from typing import List
import torch
import deepinvhessian.scalar_adcig
from pylops.signalprocessing.chirpradon2d import ChirpRadon2D

# def compute_rtm_image(model, data, dx, dt, source_amplitudes, s_cor, r_cor, device):
#     nz, nx = model.shape
#     nt = data.shape[-1]
#     num_shots = data.shape[0]
#     rtm_image = np.zeros((nz, nx), np.float32)
#     s_wavefield, r_wavefield = np.zeros((nt, nz, nx), np.float32), np.zeros((nt, nz, nx), np.float32)
#     for it in range(num_shots):
#         print(f'Shot: {it+1} / {num_shots}', flush=True)
#         s_wavefield_, r_wavefield_ = scalar_adcig.get_src_rcv(
#                 model,
#                 dx,
#                 dt,
#                 data[it::num_shots],
#                 source_amplitudes=source_amplitudes[it::num_shots].to(device),
#                 source_locations=s_cor[it::num_shots].to(device),
#                 receiver_locations=r_cor[it::num_shots].to(device),
#                 pml_width=[20, 20, 20, 20],
#                 accuracy=8,
#             )
#         # Transfer data from GPU to CPU after all GPU operations are done
#         s_wavefield[:, :, :] = s_wavefield_[:, 0, 24:-24, 24:-24].squeeze().cpu().numpy().astype(np.float32)
#         r_wavefield[:, :, :]  = r_wavefield_[:, 0, 24:-24, 24:-24].squeeze().cpu().numpy().astype(np.float32)
#         del s_wavefield_, r_wavefield_
#         torch.cuda.empty_cache()
#         gc.collect()
#         image = (s_wavefield * r_wavefield).sum(0)
#         rtm_image += image

#     return rtm_image

def compute_rtm_image(
    model: torch.Tensor,
    data: torch.Tensor,
    dx: float,
    dt: float,
    source_amplitudes: torch.Tensor,
    s_cor: torch.Tensor,
    r_cor: torch.Tensor,
    device: str
) -> np.ndarray:
    """
    Computes an RTM image using provided model, seismic data, and acquisition geometry, with all data as PyTorch tensors,
    and returns the result as a NumPy array.

    Parameters:
    - model (torch.Tensor): 2D tensor representing the velocity model with shape (nz, nx),
      where nz and nx are the number of depth and lateral samples, respectively.
    - data (torch.Tensor): 3D tensor containing the seismic data with shape (num_shots,num_receivers, nt),
      where num_shots is the number of shots, num_receivers is the number of receivers per shot, and nt is the number of time samples. 
    - dx (float): Spatial sampling interval in the lateral direction.
    - dt (float): Temporal sampling interval.
    - source_amplitudes (torch.Tensor): Tensor containing the source amplitudes for each shot.
    - s_cor (torch.Tensor): Tensor containing the source locations (coordinates) for each shot.
    - r_cor (torch.Tensor): Tensor containing the receiver locations (coordinates) for each shot.
    - device (str): String specifying the device to perform computations on, e.g., 'cpu' or 'cuda:0'.
    
    Returns:
    - rtm_image (np.ndarray): 2D NumPy array representing the computed RTM image with shape (nz, nx),
      accumulated over all shots.

    The function computes the RTM image by simulating wavefield propagation for each shot using
    the provided velocity model, and then cross-correlating the source and receiver wavefields
    to accumulate the RTM image. It utilizes PyTorch for computations to leverage GPU acceleration
    and converts the final RTM image tensor to a NumPy array before returning.
    """
    nz, nx = model.shape
    nt = data.shape[-1]
    num_shots = data.shape[0]
    rtm_image = np.zeros((nz, nx), np.float32)
    s_wavefield, r_wavefield = np.zeros((nt, nz, nx), np.float32), np.zeros((nt, nz, nx), np.float32)
    
    for it in range(num_shots):
        print(f'Shot: {it+1} / {num_shots}', flush=True)
        # Get the source and receiver wavefields
        s_wavefield_, r_wavefield_ = scalar_adcig.get_src_rcv(
                model,
                dx,
                dt,
                data[it].unsqueeze(0).to(device),
                source_amplitudes=source_amplitudes[it::num_shots].to(device),
                source_locations=s_cor[it::num_shots].to(device),
                receiver_locations=r_cor[it::num_shots].to(device),
                pml_width=[20, 20, 20, 20],  # PML (Perfectly Matched Layer) width for boundary absorption
                accuracy=8,  # Accuracy setting for the wavefield simulation
            )
        # Transfer wavefield data from GPU to CPU and remove the padding
        s_wavefield[:, :, :] = s_wavefield_[:, 0, 24:-24, 24:-24].squeeze().cpu().numpy().astype(np.float32)
        r_wavefield[:, :, :] = r_wavefield_[:, 0, 24:-24, 24:-24].squeeze().cpu().numpy().astype(np.float32)
        del s_wavefield_, r_wavefield_  # Free up GPU memory
        torch.cuda.empty_cache()  # Clear unused memory from CUDA
        gc.collect()  # Collect garbage in Python memory management
        
        # Cross-correlate source and receiver wavefields and sum over time to contribute to the RTM image
        image = (s_wavefield * r_wavefield).sum(0)
        rtm_image += image

    return rtm_image


# def compute_image_cube(s_wavefield, r_wavefield, nh, x_loc, image_cube):
    
#     _, _, nx = s_wavefield.shape
#     for i, ix in enumerate(x_loc):
#         for ih in range(-nh, nh+1):
#             if 0 <= ix+ih < nx and 0 <= ix-ih < nx:
#                 a = np.sum(s_wavefield[:, :, ix+ih] * r_wavefield[:, :, ix-ih], axis=0)
#                 image_cube[ih+nh, :, i] += a
            
#     return image_cube


# def compute_extended_images(model, data, dx, dt, source_amplitudes, s_cor, r_cor, x_loc, device):
#     nz, nx = model.shape
   
#     nh = 51
#     nt = data.shape[-1]
#     num_shots = data.shape[0]
#     image_cube = np.zeros((nh*2+1, nz, len(x_loc)), np.float32)
    
#     s_wavefield, r_wavefield = np.zeros((nt, nz, nx), np.float32), np.zeros((nt, nz, nx), np.float32)
#     for it in range(num_shots):
#         s_wavefield_, r_wavefield_ = scalar_adcig.get_src_rcv(
#                 model,
#                 dx,
#                 dt,
#                 data[it::num_shots],
#                 source_amplitudes=source_amplitudes[it::num_shots].to(device),
#                 source_locations=s_cor[it::num_shots].to(device),
#                 receiver_locations=r_cor[it::num_shots].to(device),
#                 pml_width=[20, 20, 20, 20],
#                 accuracy=8,
#             )
        
#         # Transfer data from GPU to CPU after all GPU operations are done
#         s_wavefield[:, :, :] = s_wavefield_[:, 0, 24:-24, 24:-24].squeeze().cpu().numpy().astype(np.float32)
#         r_wavefield[:, :, :]  = r_wavefield_[:, 0, 24:-24, 24:-24].squeeze().cpu().numpy().astype(np.float32)
#         del s_wavefield_, r_wavefield_
#         torch.cuda.empty_cache()
#         gc.collect()
#         print(f'Shot: {it+1} / {num_shots}', flush=True)
#         image_cube = compute_image_cube(s_wavefield, r_wavefield, nh, x_loc, image_cube)
#     return image_cube

def compute_image_cube(s_wavefield: np.ndarray, r_wavefield: np.ndarray, nh: int, x_loc: List[int], image_cube: np.ndarray) -> np.ndarray:
    """
    Computes contributions to the image cube from given source and receiver wavefields for specified crossline locations.

    Parameters:
    - s_wavefield (np.ndarray): The source wavefield as a 3D numpy array with shape (nt, nz, nx),
      where nt is the number of time samples, nz is the number of depth samples, and nx is the number of crossline samples.
    - r_wavefield (np.ndarray): The receiver wavefield as a 3D numpy array with the same shape as s_wavefield.
    - nh (int): Half the size of the horizontal aperture (offset) used for computing the image cube, defining the range [-nh, nh].
    - x_loc (List[int]): List of crossline indices for which the extended images are computed.
    - image_cube (np.ndarray): A 3D numpy array to store the resulting image cube, initialized with zeros,
      with shape (2*nh+1, nz, len(x_loc)), where the first dimension covers offsets from -nh to nh.

    Returns:
    - image_cube (np.ndarray): The updated image cube with contributions added from the current set of source and
      receiver wavefields for the specified crossline locations.

    The function iterates over each specified crossline location and offset within the provided aperture, computing
    the cross-correlation of the source and receiver wavefields at each offset and adding the result to the image cube.
    """
    _, _, nx = s_wavefield.shape
    for i, ix in enumerate(x_loc):
        for ih in range(-nh, nh+1):
            if 0 <= ix+ih < nx and 0 <= ix-ih < nx:
                image = np.sum(s_wavefield[:, :, ix+ih] * r_wavefield[:, :, ix-ih], axis=0)
                image_cube[ih+nh, :, i] += image
            
    return image_cube


def compute_extended_images(model: np.ndarray, data: np.ndarray, dx: float, dt: float, source_amplitudes: torch.Tensor, s_cor: torch.Tensor, r_cor: torch.Tensor, x_loc: List[int], device: str) -> np.ndarray:
    """
    Computes extended images using a velocity model, seismic data, source amplitudes, source and receiver locations,
    and specified crossline locations.

    Parameters:
    - model (np.ndarray): The velocity model as a 2D numpy array with shape (nz, nx),
      where nz is the number of depth samples and nx is the number of crossline samples.
    - data (np.ndarray): The seismic data as a 3D numpy array with shape (num_shots, num_receivers, nt),
      where num_shots is the number of shots, num_receivers is the number of receivers per shot, and nt is the number of time samples.
    - dx (float): Spatial sampling interval in the lateral direction.
    - dt (float): Temporal sampling interval.
    - source_amplitudes (torch.Tensor): Source amplitudes for each shot as a PyTorch tensor.
    - s_cor (torch.Tensor): Source locations (coordinates) for each shot as a PyTorch tensor.
    - r_cor (torch.Tensor): Receiver locations (coordinates) for each shot as a PyTorch tensor.
    - x_loc (List[int]): List of crossline indices for which the extended images are computed.
    - device (str): The computing device (e.g., 'cpu' or 'cuda:0').

    Returns:
    - image_cube (np.ndarray): A 3D numpy array representing the computed extended images with shape (2*nh+1, nz, len(x_loc)),
      where the first dimension covers offsets from -nh to nh, nz is the number of depth samples, and len(x_loc) is the
      number of specified crossline locations.

    The function simulates wavefield propagation for each shot based on the provided velocity model and seismic data,
    then computes contributions to the extended images for specified crossline locations by correlating the source and
    receiver wavefields at different offsets. The result is accumulated in an image cube.
    """
    nz, nx = model.shape
    nh = 51 
    nt = data.shape[-1]
    num_shots = data.shape[0]
    image_cube = np.zeros((nh*2+1, nz, len(x_loc)), np.float32)
    
    s_wavefield, r_wavefield = np.zeros((nt, nz, nx), np.float32), np.zeros((nt, nz, nx), np.float32)
    for it in range(num_shots):
        s_wavefield_, r_wavefield_ = scalar_adcig.get_src_rcv(
                model,
                dx,
                dt,
                data[it::num_shots],
                source_amplitudes=source_amplitudes[it::num_shots].to(device),
                source_locations=s_cor[it::num_shots].to(device),
                receiver_locations=r_cor[it::num_shots].to(device),
                pml_width=[20, 20, 20, 20],
                accuracy=8,
            )
        
        # Transfer data from GPU to CPU after all GPU operations are done
        s_wavefield[:, :, :] = s_wavefield_[:, 0, 24:-24, 24:-24].squeeze().cpu().numpy().astype(np.float32)
        r_wavefield[:, :, :]  = r_wavefield_[:, 0, 24:-24, 24:-24].squeeze().cpu().numpy().astype(np.float32)
        del s_wavefield_, r_wavefield_
        torch.cuda.empty_cache()
        gc.collect()
        print(f'Shot: {it+1} / {num_shots}', flush=True)
        image_cube = compute_image_cube(s_wavefield, r_wavefield, nh, x_loc, image_cube)
    return image_cube



def angle_gather(image: np.ndarray, velocity: np.ndarray, dx: float, alpha_max: float, ixs: np.ndarray = None) -> np.ndarray:
    """
    Calculate the Angle Domain Common Image Gather (ADCIG) from a seismic extended image for specified crossline indices or all crosslines.

    Parameters:
    - image (np.ndarray): A 3D numpy array of shape (nh, nz, nx) representing the seismic extended image,
      where nh is the number of offsets, nz is the number of depth samples, and nx is the number of crossline samples
      (either total crosslines or specific crosslines, depending on the context).
    - velocity (np.ndarray): A 2D numpy array of shape (nz, nx) representing the velocity model,
      with dimensions matching the depth and crossline dimensions of the image.
    - dx (float): The spatial sampling interval in the crossline direction.
    - alpha_max (float): The maximum angle (in degrees) to be considered in the ADCIG.
    - ixs (np.ndarray, optional): An array of crossline indices for which the ADCIG will be calculated.
      If None (default), the ADCIG is calculated for all crossline samples available in the image.

    Returns:
    - ang_gather (np.ndarray): A 3D numpy array of shape (number_of_processed_crosslines, nz, nalpha),
      representing the ADCIG. The data type of the array is np.float32. The number_of_processed_crosslines corresponds
      to the length of ixs if provided, or nx from the image if ixs is None.

    The function computes the ADCIG by first performing a Slant Stack or Radon Transform on the seismic extended image,
    then interpolating the transformed data into angle bins. The angle bins range from -alpha_max to alpha_max,
    allowing for analysis of seismic events at various angles of incidence.
    """
    # Extract dimensions from the image
    nh, nz, nx = image.shape
    
    # Define the time (depth) and offset axes for the Radon transform
    taxis = np.arange(nz)
    xaxis = np.arange(-nh//2, nh//2)
    
    # Determine minimum velocity and compute px, the slowness sampling interval
    vel_min = velocity.min()
    px = vel_min * dx * 1e-4
    
    # Define the slowness axis for the transform
    paxis = np.arange(-nh//2, nh//2) * px * 1e-5
    
    # Initialize the slant stack array
    ssk = np.zeros((nx, nz, nh))
    
    # Calculate the number of angle samples and define the angle axis
    nalpha = alpha_max * 10
    alpha = np.linspace(-alpha_max, alpha_max, nalpha)
    
    # Initialize the output ADCIG array
    ang_gather = np.zeros((nx, nz ,nalpha), dtype=np.float32)
    
    # Initialize the Slant Stack (Radon Transform) operator
    SlantStack = ChirpRadon2D(taxis, xaxis, px, dtype="float64")
    if ixs is None: ixs = np.arange(nx)
    # Loop over crossline samples
    for ix, ixx in enumerate(ixs):
        # Perform Slant Stack on each crossline sample of the image
        res = SlantStack * image[:,:,ix].ravel()
        ssk[ix] = res.reshape(nh, nz).T
        
        # Loop over depth samples
        for iz in np.arange(nz):
            pxs = ssk[ix][iz]
            
            # Calculate the incidence angles for each depth sample based on velocity model

            theta = np.arcsin(paxis * velocity[iz, ixx]) * 180 / np.pi
            
            # Interpolate slant stack results into angle bins
            ang_gather[ix][iz] = np.interp(alpha, theta, pxs)
    
    # Replace NaN values with zeros
    ang_gather[np.isnan(ang_gather)] = 0
    
    return ang_gather