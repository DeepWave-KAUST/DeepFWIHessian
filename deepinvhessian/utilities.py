import random
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
import torch


def cosine_taper(nz: int, nx: int, top_width: int = 10, bottom_width: int = 10) -> np.ndarray:
    """
    Generate a 1D cosine taper with separate control over the top and bottom widths, 
    and repeat it horizontally across nx points.

    This function creates a 1D cosine taper of size `nz` where the taper values 
    are repeated horizontally across `nx` columns, effectively applying the same 
    taper to each column. The widths of the taper at the top and bottom can be 
    controlled separately.

    Parameters:
    - nz (int): Number of points in the vertical direction where taper is applied.
    - nx (int): Number of points in the horizontal direction to repeat the taper.
    - top_width (int, optional): The width of the taper on the top side. Default is 10.
    - bottom_width (int, optional): The width of the taper on the bottom side. Default is 10.

    Returns:
    - np.ndarray: A 2D array of size (nz, nx) with the cosine taper applied vertically.

    Examples:
    >>> taper = cosine_taper(100, 50, top_width=20, bottom_width=10)
    >>> taper.shape
    (100, 50)
    """
    taper_z = np.ones(nz)

    # Apply tapering to the top
    for i in range(top_width):
        scale = 0.5 * (1 - np.cos(np.pi * i / top_width))
        taper_z[i] = scale

    # Apply tapering to the bottom
    for i in range(bottom_width):
        scale = 0.5 * (1 - np.cos(np.pi * i / bottom_width))
        taper_z[-(i+1)] = scale

    # Repeat the taper along the horizontal direction
    taper = np.tile(taper_z[:, np.newaxis], (1, nx))

    return taper


def cosine_taper1d(nz: int, top_width: int = 10, bottom_width: int = 10) -> np.ndarray:
    """
    Create a 1D cosine taper.

    :param nz: int
        Length of the array to apply the taper to.
    :param top_width: int, optional
        Width of the taper at the start (top) of the array, default is 10.
    :param bottom_width: int, optional
        Width of the taper at the end (bottom) of the array, default is 10.
    :return: np.ndarray
        The tapered array.
    """
    taper_z = np.ones(nz)

    # Apply tapering to the top
    for i in range(top_width):
        scale = 0.5 * (1 - np.cos(np.pi * i / top_width))
        taper_z[i] = scale

    # Apply tapering to the bottom
    for i in range(bottom_width):
        scale = 0.5 * (1 - np.cos(np.pi * i / bottom_width))
        taper_z[-(i+1)] = scale

    return taper_z


def show_model(
    X: ArrayLike,
    *,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None, 
    extent: tuple[float, float, float, float] | None = None,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (16, 10),
    save_path: str | None = None,
):
    """
    Display a model using an image plot.

    Parameters:
    -----------
    X : ArrayLike
        The 2D array representation of the model to be plotted.

    cmap : str, optional
        The colormap used to map normalized data values to RGBA colors. Defaults to None.

    vmin, vmax : float, optional
        The colorbar range. If either is None, the min and max of the data will be used. Defaults to None.

    extent : tuple of float, optional
        Bounding box in data coordinates that the image will fill (left, right, bottom, top). Defaults to None.

    title : str, optional
        Title to display on the plot. Defaults to None.

    xlim, ylim : tuple of float, optional
        The x and y axis limits (min, max). Defaults to None.

    figsize : tuple of float, optional
        The size of the figure (width, height). Defaults to (16, 10).

    save_path : str, optional
        If provided, the plot will be saved to this path. The plot is saved in high resolution (300 dpi). Defaults to None.

    Returns:
    --------
    None

    Example:
    --------
    >>> show_model(my_model, cmap='gray', title='My Model', xlim=(0, 100), ylim=(0, 100))
    """
    if vmin==None or vmax==None:
        vmin, vmax = np.percentile(X, [2,98])
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    image = ax.imshow(X, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
    ax.set_xlabel(r'X [m]')
    ax.set_ylabel(r'Z [m]')
    ax.set_title(title)
    ax.axis('tight')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    fig.colorbar(image, pad=0.01)
    if save_path is not None:
        plt.savefig(f'{save_path}', bbox_inches='tight', dpi=300)
    plt.show()

def show_3_shots(
    X: ArrayLike,
    shots: tuple[int, int, int],
    *,
    cmap: str | None = 'gray',
    vmin: float | None = None,
    vmax: float | None = None, 
    extent: tuple[float, float, float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    clip: float=1.0,
    figsize: tuple[float, float] = (16, 9),
    save_path: str | None = None,
):
    """
    Display 3 seismic shot gathers side-by-side.

    Parameters:
    -----------
    X : ArrayLike
        A 3D array representation of the seismic data arranged as shots x receivers x time.

    shots : tuple of int
        The indices of the 3 shot gathers to be displayed.

    cmap : str, optional
        The colormap used to map normalized data values to RGBA colors. Defaults to 'gray'.

    vmin, vmax : float, optional
        The colorbar range. If either is None, they will be determined based on the clip factor and data amplitude. Defaults to None.

    extent : tuple of float, optional
        Bounding box in data coordinates that the image will fill (left, right, bottom, top). Defaults to None.

    title : str, optional
        Title to display on the plot. Currently, this parameter is unused in the function. Defaults to None.

    xlim, ylim : tuple of float, optional
        The x and y axis limits (min, max) for the displayed shots. Currently, only ylim is utilized in the function. Defaults to None.

    clip : float, optional
        Factor to determine the vmax based on the maximum absolute amplitude of the data. Defaults to 1.0.

    figsize : tuple of float, optional
        The size of the figure (width, height). Defaults to (16, 9).

    save_path : str, optional
        If provided, the plot will be saved to this path. The plot is saved in high resolution (300 dpi). Defaults to None.

    Returns:
    --------
    None

    Example:
    --------
    >>> show_3_shots(data, shots=(10, 20, 30), cmap='gray', clip=0.8, ylim=(0, 150))
    """

    if vmin is None or vmax is None:
        vmax = clip * np.abs(X).max()
        vmin = -vmax

    fig, axs = plt.subplots(1, 3, sharey=True, figsize=figsize)
    axs[0].imshow(X[shots[0]].T, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
    axs[0].set_title(f'Shot gather {shots[0]+1}')
    axs[0].set_xlabel(r'X [km]')
    axs[0].axis('tight')
    axs[0].set_ylabel(r'Time [s]')
    axs[0].set_xlim(xlim)
    axs[1].imshow(X[shots[1]].T, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
    axs[1].set_title(f'Shot gather {shots[1]+1}')
    axs[1].set_xlabel(r'X [km]')
    axs[1].axis('tight')
    axs[1].set_xlim(xlim)
    axs[2].imshow(X[shots[2]].T, cmap=cmap, vmin=vmin, vmax=vmax, extent=extent)
    axs[2].set_title(f'Shot gather {shots[2]+1}')
    axs[2].set_xlabel(r'X [km]')
    axs[2].axis('tight')
    axs[2].set_xlim(xlim)
    axs[2].set_ylim(ylim)
    if save_path is not None:
        plt.savefig(f'{save_path}', bbox_inches='tight', dpi=300)
    plt.show()
    
    # https://github.com/swag-kaust/Transform2022_SelfSupervisedDenoising/blob/main/tutorial_utils.py
def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    
    Parameters
    ----------
    seed: int 
        Integer to be used for the seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True

def show_one_iter_fwi(
    grad_fwi: ArrayLike,
    update: ArrayLike,
    iteration: int = 0,
    *,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None, 
    extent: tuple[float, float, float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (16, 4),
    save_path: str | None = None,
  )-> None:
    """
    Visualizes the gradient of the Full Waveform Inversion (FWI) and the updated model for a given iteration.

    This function creates two subplots: the first displaying the gradient of the FWI and the second showing the updated model.
    The color scales can be adjusted, and the images can be saved to a specified path.

    Parameters
    ----------
    grad_fwi : ArrayLike
        The calculated FWI gradient in the current iteration.
    update : ArrayLike
        The update to the model as calculated in the current iteration.
    iteration : int, optional
        The iteration number of the FWI process (default is 0).
    cmap : str or None, optional
        The colormap for the gradient plot (default is None, which will use the default matplotlib colormap).
    vmin : float or None, optional
        The minimum value for color scaling the updated model plot (default is None, autoscales to the data).
    vmax : float or None, optional
        The maximum value for color scaling the updated model plot (default is None, autoscales to the data).
    extent : tuple of four floats or None, optional
        The bounding box in data coordinates that the image will fill specified as (left, right, bottom, top) in data coordinates (default is None).
    xlim : tuple of two floats or None, optional
        The limit of the x-axis (default is None, which autoscales to the x extent of the data).
    ylim : tuple of two floats or None, optional
        The limit of the y-axis (default is None, which autoscales to the y extent of the data).
    figsize : tuple of two floats
        The figure dimension (width, height) in inches (default is (16, 4)).
    save_path : str or None, optional
        The directory path to save the figure. If None, the figure is not saved (default is None).

    Returns
    -------
    None
        This function does not return a value; it creates and shows a matplotlib figure.
    """

    r = 0.95
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    m_min1, m_max1 = np.percentile(grad_fwi, [2,98])
    scale = np.abs([m_min1, m_max1]).max()

    img1 = axs[0].imshow(grad_fwi, vmin=-scale, vmax=scale, cmap=cmap, 
                           interpolation='bilinear', extent=extent)
    plt.colorbar(img1, ax=axs[0], pad=0.02, shrink=r)
    axs[0].set_title(f'FWI Gradient', fontsize=14)
    axs[0].set_xlabel(r'X [m]')
    axs[0].set_ylabel(r'Z [m]')
    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)

    img2 = axs[1].imshow(update, vmin=vmin, vmax=vmax, cmap='jet', 
                           interpolation='bilinear', extent=extent)
    plt.colorbar(img2, ax=axs[1], pad=0.02, shrink=r)
    axs[1].set_title('Updated Model', fontsize=14)
    axs[1].set_xlabel(r'X [m]')
    axs[1].set_ylabel(r'Z [m]')
    axs[1].set_xlim(xlim)
    axs[1].set_ylim(ylim)
    
    plt.suptitle(f'Iteration {iteration+1}', x=0.5, y=0.95, fontsize=16)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f'{save_path}/grad_iter_{iteration+1}.png',  bbox_inches='tight', dpi=300)
    plt.show()

def show_one_iter_dm(
    grad_fwi: ArrayLike,
    dm1: ArrayLike,
    grad_pred: ArrayLike,
    dm: ArrayLike,
    update: ArrayLike,
    network_loss: ArrayLike,
    iteration: int = 0,
    *,
    cmap: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None, 
    extent: tuple[float, float, float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (12, 12),
    save_path: str | None = None,
  )-> None:
    """
    Visualizes various components of an iteration during the inversion process including 
    FWI gradients, predicted updates, and network loss over iterations.

    The function generates a 3x2 grid of plots displaying the gradient of the FWI, the migrated image using 
    the gradient, the predicted gradient, the predicted model update from gradient, the final
    model update, and the network loss per iteration.

    Parameters
    ----------
    grad_fwi : ArrayLike
        The calculated gradient of the FWI.
    dm1 : ArrayLike
        The migrated image using the gradient.
    grad_pred : ArrayLike
        The predicted gradient from the neural network.
    dm : ArrayLike
        The predicted model update from gradient.
    update : ArrayLike
        The final model update applied to the model in the current iteration.
    network_loss : ArrayLike
        A sequence of loss values obtained from the network training over iterations.
    iteration : int, optional
        The iteration number of the inversion process (default is 0).
    cmap : Optional[str], optional
        The colormap for the gradient and update plots (default is None, which uses the default colormap).
    vmin : Optional[float], optional
        The minimum value for color scaling in the final update plot (default is None, autoscales to the data).
    vmax : Optional[float], optional
        The maximum value for color scaling in the final update plot (default is None, autoscales to the data).
    extent : Optional[Tuple[float, float, float, float]], optional
        The bounding box in data coordinates that the image will fill, specified as (left, right, bottom, top) (default is None).
    xlim : Optional[Tuple[float, float]], optional
        The limit of the x-axis for the gradient and update plots (default is None, which autoscales to the x extent of the data).
    ylim : Optional[Tuple[float, float]], optional
        The limit of the y-axis for the gradient and update plots (default is None, which autoscales to the y extent of the data).
    figsize : Tuple[float, float]
        The figure dimension (width, height) in inches (default is (12, 12)).
    save_path : Optional[str], optional
        The directory path where the figure will be saved. If None, the figure is not saved (default is None).

    Returns
    -------
    None
        The function does not return a value; instead, it shows or saves a matplotlib figure.
    """


    r = 0.81
    fig, axs = plt.subplots(3, 2, figsize=figsize)

    m_min1, m_max1 = np.percentile(grad_fwi, [2,98])
    m_min2, m_max2 = np.percentile(dm1, [2,98])
    m_min3, m_max3 = np.percentile(grad_pred, [2,98])
    m_min4, m_max4 = np.percentile(dm, [2,98])
    scale = np.abs([m_min1, m_max1, m_min2, m_max2, m_min3, m_max3, m_min4, m_max4]).max()

    img1 = axs[0,0].imshow(grad_fwi, vmin=-scale, vmax=scale, cmap=cmap, 
                           interpolation='bilinear', extent=extent)
    plt.colorbar(img1, ax=axs[0,0], pad=0.02, shrink=r)
    axs[0,0].set_title(f'FWI Gradient', fontsize=14)
    axs[0,0].set_xlabel(r'X [m]')
    axs[0,0].set_ylabel(r'Z [m]')
    axs[0,0].set_xlim(xlim)
    axs[0,0].set_ylim(ylim)

    img2 = axs[0,1].imshow(dm1, vmin=-scale, vmax=scale, cmap=cmap, 
                           interpolation='bilinear', extent=extent)
    plt.colorbar(img2, ax=axs[0,1], pad=0.02, shrink=r)
    axs[0,1].set_title(r'$\delta m_1$', fontsize=14)
    axs[0,1].set_xlabel(r'X [m]')
    axs[0,1].set_ylabel(r'Z [m]')
    axs[0,1].set_xlim(xlim)
    axs[0,1].set_ylim(ylim)
    
    img3 = axs[1,0].imshow(grad_pred, vmin=-scale, vmax=scale, cmap=cmap, 
                           interpolation='bilinear', extent=extent)
    fig.colorbar(img3, ax=axs[1,0], pad=0.02, shrink=r)
    axs[1,0].set_title(r'$f_{\theta}(\delta m_1)$', fontsize=14)
    axs[1,0].set_xlabel(r'X [m]')
    axs[1,0].set_ylabel(r'Z [m]')
    axs[1,0].set_xlim(xlim)
    axs[1,0].set_ylim(ylim)

    img4 = axs[1,1].imshow(dm, vmin=-scale, vmax=scale, cmap=cmap, 
                           interpolation='bilinear', extent=extent)
    fig.colorbar(img4, ax=axs[1,1], pad=0.02, shrink=r)
    axs[1,1].set_title(r'$f_{\theta}(gradient)$', fontsize=14)
    axs[1,1].set_xlabel(r'X [m]')
    axs[1,1].set_ylabel(r'Z [m]')
    axs[1,1].set_xlim(xlim)
    axs[1,1].set_ylim(ylim)

    img5 = axs[2,0].imshow(update, vmin=vmin, vmax=vmax, cmap='jet', 
                           interpolation='bilinear', extent=extent)
    plt.colorbar(img5, ax=axs[2,0], pad=0.02, shrink=r)
    axs[2,0].set_title('Updated Model', fontsize=14)
    axs[2,0].set_xlabel(r'X [m]')
    axs[2,0].set_ylabel(r'Z [m]')
    axs[2,0].set_xlim(xlim)
    axs[2,0].set_ylim(ylim)

    axs[2,1].plot(network_loss)
    axs[2,1].set_title('Network loss')
    axs[2,1].set_xlabel('Iteration')
    
    plt.suptitle(f'Iteration {iteration+1}', x=0.5, y=0.92, fontsize=16)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f'{save_path}/dms_iter_{iteration+1}.png',  bbox_inches='tight', dpi=300)
    plt.show()
