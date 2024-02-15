import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
import scipy as sp
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def lowpass_filter(order: int, freq: float, data: ArrayLike, dt: float, filteringN: int = 1) -> ArrayLike:
    """
    Apply a lowpass filter to the input data.

    Parameters:
    -----------
    order : int
        The order of the filter.
    freq : float
        The cutoff frequency (in Hz) of the filter.
    data : array-like
        The input data to be filtered.
    dt : float
        Time step interval (in seconds) of the input data.
    filteringN : int, optional
        If 1, applies forward filtering. If 2, applies forward and backward filtering. Defaults to 1.

    Returns:
    --------
    array-like
        The filtered data.
    """
    sos = sp.signal.butter(N=order, Wn=freq, btype='lowpass', output='sos', fs=1/dt)
    if filteringN == 1:
        return sp.signal.sosfilt(sos, data, axis=-1).copy()
    return sp.signal.sosfiltfilt(sos, data, axis=-1).copy()

def highpass_filter(order: int, freq: float, data: ArrayLike, dt: float, filteringN: int = 1) -> ArrayLike:
    """
    Apply a highpass filter to the input data.

    Parameters:
    -----------
    order : int
        The order of the filter.
    freq : float
        The cutoff frequency (in Hz) of the filter.
    data : array-like
        The input data to be filtered.
    dt : float
        Time step interval (in seconds) of the input data.
    filteringN : int, optional
        If 1, applies forward filtering. If 2, applies forward and backward filtering. Defaults to 1.

    Returns:
    --------
    array-like
        The filtered data.
    """
    sos = sp.signal.butter(order, Wn=freq, btype='highpass', output='sos', fs=1/dt)
    if filteringN == 1:
        return sp.signal.sosfilt(sos, data, axis=-1).copy()
    return sp.signal.sosfiltfilt(sos, data, axis=-1).copy()

def bandpass_filter(order: int, low: float, high: float, data: ArrayLike, dt: float, filteringN: int = 1) -> ArrayLike:
    """
    Apply a bandpass filter to the input data.

    Parameters:
    -----------
    order : int
        The order of the filter.
    low, high : float
        The lower and upper frequency bounds (in Hz) for the bandpass filter.
    data : array-like
        The input data to be filtered.
    dt : float
        Time step interval (in seconds) of the input data.
    filteringN : int, optional
        If 1, applies forward filtering. If 2, applies forward and backward filtering. Defaults to 1.

    Returns:
    --------
    array-like
        The filtered data.
    """
    sos = sp.signal.butter(order, Wn=[low, high], btype='bandpass', output='sos', fs=1/dt)
    if filteringN == 1:
        return sp.signal.sosfilt(sos, data, axis=-1).copy()
    return sp.signal.sosfiltfilt(sos, data, axis=-1).copy()

def gaussian_kernel(size: int, sigma: float) -> Tensor:
    """
    Creates a 1D Gaussian kernel using PyTorch.

    Parameters
    ----------
    size : int
        The size of the kernel. This is the number of elements in the returned tensor.
    sigma : float
        The standard deviation of the Gaussian distribution. This controls the spread of the kernel.

    Returns
    -------
    Tensor
        A 1D tensor of shape `(size,)` representing the Gaussian kernel. The kernel is normalized so that its sum equals 1.

    Example
    -------
    >>> gaussian_kernel(5, 1.0)
    tensor([0.0618, 0.2448, 0.3873, 0.2448, 0.0618])
    """
    x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
    gaussian = torch.exp(-x**2 / (2 * sigma**2))
    return gaussian / gaussian.sum()

def apply_1d_gaussian_filter(
    tensor: Tensor, 
    sigma_h: float, 
    sigma_v: float, 
    kernel_size: int = None, 
    device: str = 'cpu'
) -> Tensor:
    """
    Applies a 1D Gaussian filter separately in the horizontal and vertical directions to a 2D tensor.

    Parameters
    ----------
    tensor : Tensor
        The input 2D tensor to which the Gaussian filter will be applied.
    sigma_h : float
        Standard deviation of the Gaussian kernel for horizontal filtering.
    sigma_v : float
        Standard deviation of the Gaussian kernel for vertical filtering.
    kernel_size : int, optional
        Size of the Gaussian kernel. If not specified, it is calculated based on the sigma values.
        Ensures the kernel size is odd so the filter has a center.
    device : str, optional
        The device to perform the computation on. Default is 'cpu', can be set to 'cuda' for GPU computation.

    Returns
    -------
    Tensor
        The filtered 2D tensor after applying the Gaussian filter in both directions.

    Example
    -------
    >>> tensor = torch.rand(10, 10)
    >>> filtered_tensor = apply_1d_gaussian_filter(tensor, sigma_h=1.0, sigma_v=1.5, kernel_size=5, device='cpu')
    """
    # Ensure the tensor is on the specified device
    tensor = tensor.to(device)

    # Calculate kernel sizes if not provided
    if kernel_size is None:
        kernel_size_h = int(np.ceil(sigma_h * 6))  # to cover +/- 3 standard deviations for horizontal
        if kernel_size_h % 2 == 0:
            kernel_size_h += 1  # ensuring size is odd

        kernel_size_v = int(np.ceil(sigma_v * 6))  # to cover +/- 3 standard deviations for vertical
        if kernel_size_v % 2 == 0:
            kernel_size_v += 1  # ensuring size is odd
    else:
        kernel_size_h = kernel_size_v = kernel_size

    # Creating the Gaussian kernels and moving them to the specified device
    gaussian_kernel_1d_h = gaussian_kernel(kernel_size_h, sigma_h).view(1, 1, kernel_size_h, 1).to(device)
    gaussian_kernel_1d_v = gaussian_kernel(kernel_size_v, sigma_v).view(1, 1, 1, kernel_size_v).to(device)

    # Apply the kernel in the horizontal direction
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    filtered_tensor_h = F.conv2d(tensor, gaussian_kernel_1d_h, padding=(kernel_size_h // 2, 0))

    # Apply the kernel in the vertical direction
    filtered_tensor_v = F.conv2d(filtered_tensor_h, gaussian_kernel_1d_v, padding=(0, kernel_size_v // 2))

    return filtered_tensor_v.squeeze(0).squeeze(0)  # Remove batch and channel dimensions
