import gc
import torch
import numpy as np
import deepwave


class FWIParams:
    def __init__(self, par: dict, acquisition: int = 1) -> None:
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
        self.nx: int = par['nx']
        self.nz: int = par['nz']
        self.dx: float = par['dx']
        self.nt: int = par['nt']
        self.dt: float = par['dt']
        self.num_dims: int = par['num_dims']
        self.num_shots: int = par['ns']
        self.num_batches: int = par['num_batches']
        self.num_sources_per_shot: int = 1
        self.num_receivers_per_shot: int = par['nr']
        self.ds: float = par['ds']
        self.dr: float = par['dr']
        self.sz: float = par['sz']
        self.rz: float = par['rz']
        self.os: float = par['osou']
        self.orec: float = par['orec']
        self.ox: float = par['ox']
        self.freq: float = par['freq']
        self.s_cor, self.r_cor = self.get_coordinate(acquisition)

    def get_coordinate(self, mode: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the coordinates of sources and receivers.

        Parameters:
        - mode (int): Type of acquisition (1 or 2)
            1: Receivers are spreaded over the whole surface, 
            2: Specific offset

        Returns:
        - s_cor (torch.Tensor): Coordinates of sources.
        - r_cor (torch.Tensor): Coordinates of receivers.

        """
        x_s = torch.zeros(self.num_shots, self.num_sources_per_shot, self.num_dims)
        x_r = torch.zeros(self.num_shots, self.num_receivers_per_shot, self.num_dims)
        x_s[:, 0, 1] = torch.arange(0, self.num_shots).float() * self.ds + self.os - self.ox
        x_s[:, 0, 0] = self.sz

        if mode == 1:
            x_r[0, :, 1] = torch.arange(0, self.num_receivers_per_shot).float() * self.dr + self.orec
            x_r[:, :, 1] = x_r[0, :, 1].repeat(self.num_shots, 1)
            x_r[:, :, 0] = self.rz
        elif mode == 2:
            x_r[0, :, 1] = torch.arange(self.num_receivers_per_shot).float() * self.dr + self.orec
            x_r[:, :, 1] = x_r[0, :, 1].repeat(self.num_shots, 1) + \
                           torch.arange(0, self.num_shots).repeat(self.num_receivers_per_shot, 1).T * self.ds - self.ox
            x_r[:, :, 0] = self.rz

        return x_s, x_r

# def Ricker(params):
#    """Create a Ricker wavelet.

#    Parameters:
#    ----------
#    freq : float
#       Dominant frequency
#    Returns
#    -------
#    nd.array
#       Ricker wavelet
#    """
#    return (deepwave.wavelets.ricker(params.freq, params.nt, params.dt, 1/params.freq)
#                            .reshape(-1, 1, 1))
def Ricker(params: dict) -> torch.Tensor:
   """Create a Ricker wavelet.
   
   Parameters:
   - params (dict): Dictionary containing the parameters.
      - freq (float): Dominant frequency
      - nt (int): Number of time samples
      - dt (float): Time sampling interval 
   
   Returns:
   - wavelet (torch.Tensor): Ricker wavelet """ 
   wavelet = deepwave.wavelets.ricker(params.freq, params.nt, params.dt, 1/params.freq)
   wavelet = wavelet.reshape(-1, 1, 1) 
   
   return wavelet 


def forward_modelling(params: FWIParams, model: torch.Tensor, wavelet: torch.Tensor, device: str) -> torch.Tensor:
    """Perform 2D acoustic wave equation forward modeling.

    Parameters:
    - params (FWIParams): Object containing the parameters for forward modeling.
    - model (torch.Tensor): Model tensor.
    - wavelet (torch.Tensor): Wavelet tensor.
    - device (str): Device to perform computation on (e.g., 'cuda', 'cpu').

    Returns:
    - data (torch.Tensor): Simulated seismic data.

    """
    # pml_width parameter controls the boundary; for free surface, the first argument should be 0
    prop = deepwave.scalar.Propagator(
        {'vp': model.to(device)},
        params.dx,
        pml_width=[0, 20, 20, 20, 0, 0]
    ).to(device)
    
    data = prop(
        wavelet.to(device),
        params.s_cor.to(device),
        params.r_cor.to(device),
        params.dt
    )
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return data


# def propagator(params,model,device):
#    return deepwave.scalar.Propagator(
#       {'vp': model.to(device)}, 
#       params.dx, 
#       pml_width=[0,20,20,20,0,0]).to(device)

# def propagator_born(params, model, dm1, device):
#    return deepwave.scalar.BornPropagator({'vp': model.detach(),
#                                           'scatter': dm1}, 
#                                           params.dx, 
#                                           pml_width=[0, 20, 20, 20, 0, 0]).to(device)

# def compute_gradient(params, model, data, wavelet, optimizer, device):
#    criterion = torch.nn.MSELoss()
   
   
#    running_loss = 0 
#    num_batches = params.num_batches
#    optimizer.zero_grad()
#    for it in range(num_batches): # loop over shots
#       prop = deepwave.scalar.Propagator(
#       {'vp': model.to(device)}, 
#       params.dx, 
#       pml_width=[0,20,20,20,0,0]).to(device) 
#       batch_wavl = wavelet[:,it::num_batches].to(device)
#       batch_data_true = data[:,it::num_batches].to(device)
#       batch_x_s = params.s_cor[it::num_batches].to(device)
#       batch_x_r = params.r_cor[it::num_batches].to(device)
#       batch_data_pred = prop(batch_wavl, batch_x_s, batch_x_r, params.dt)
#       loss = criterion(batch_data_pred, batch_data_true)
#       running_loss += loss.item()
#       loss.backward()
#    gc.collect()
#    torch.cuda.empty_cache()
#    return model.grad.clone(), running_loss

def compute_gradient(params: FWIParams, model: torch.Tensor, data: torch.Tensor, wavelet: torch.Tensor,
                     optimizer: torch.optim.Optimizer, device: str) -> tuple[torch.Tensor, float]:
    """Compute the gradient of the model parameters using backpropagation.

    Parameters:
    - params (FWIParams): Object containing the parameters for computation.
    - model (torch.Tensor): Model tensor.
    - data (torch.Tensor): Observed seismic data tensor.
    - wavelet (torch.Tensor): Wavelet tensor.
    - optimizer (torch.optim.Optimizer): Optimizer for updating the model parameters.
    - device (str): Device to perform computation on (e.g., 'cuda', 'cpu').

    Returns:
    - gradient (torch.Tensor): Gradient of the model parameters.
    - running_loss (float): Accumulated loss during the computation.

    """
    criterion = torch.nn.MSELoss()
    running_loss = 0
    num_batches = params.num_batches
    
    optimizer.zero_grad()
    
    for it in range(num_batches):  # loop over shots
        prop = deepwave.scalar.Propagator(
            {'vp': model.to(device)},
            params.dx,
            pml_width=[0, 20, 20, 20, 0, 0]
        ).to(device)
        
        batch_wavl = wavelet[:, it::num_batches].to(device)
        batch_data_true = data[:, it::num_batches].to(device)
        batch_x_s = params.s_cor[it::num_batches].to(device)
        batch_x_r = params.r_cor[it::num_batches].to(device)
        
        batch_data_pred = prop(batch_wavl, batch_x_s, batch_x_r, params.dt)
        
        loss = criterion(batch_data_pred, batch_data_true)
        running_loss += loss.item()
        
        loss.backward()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return model.grad.clone(), running_loss


# def compute_dm1(params, model, dm1, wavelet, optimizer, device):
#    criterion = torch.nn.MSELoss()
#    prop = deepwave.scalar.BornPropagator({'vp': model.to(device).detach(),
#                                           'scatter': dm1.to(device)}, 
#                                           params.dx, 
#                                           pml_width=[0, 20, 20, 20, 0, 0]).to(device)
   
#    running_loss = 0 
#    num_batches = params.num_batches
#    optimizer.zero_grad()
#    for it in range(num_batches): # loop over shots 
#       batch_wavl = wavelet[:,it::num_batches].to(device)
#       batch_x_s = params.s_cor[it::num_batches].to(device)
#       batch_x_r = params.r_cor[it::num_batches].to(device)
#       batch_data_pred = prop(batch_wavl, batch_x_s, batch_x_r, params.dt)
#       loss = criterion(batch_data_pred,  torch.zeros_like(batch_data_pred))
#       running_loss += loss.item()
#       loss.backward()
#    gc.collect()
#    torch.cuda.empty_cache()
#    return dm1.grad.clone()


def compute_dm1(params: FWIParams, model: torch.Tensor, dm1: torch.Tensor, wavelet: torch.Tensor,
                optimizer: torch.optim.Optimizer, device: str) -> torch.Tensor:
    """Compute the gradient of the scattering perturbation (dm1) using Born modeling and backpropagation.

    Parameters:
    - params (FWIParams): Object containing the parameters for computation.
    - model (torch.Tensor): Model tensor.
    - dm1 (torch.Tensor): Scattering perturbation (dm1) tensor.
    - wavelet (torch.Tensor): Wavelet tensor.
    - optimizer (torch.optim.Optimizer): Optimizer for updating the dm1 tensor.
    - device (str): Device to perform computation on (e.g., 'cuda', 'cpu').

    Returns:
    - dm1_grad (torch.Tensor): Gradient of the scattering perturbation (dm1).

    """
    criterion = torch.nn.MSELoss()
    
    prop = deepwave.scalar.BornPropagator(
        {'vp': model.to(device).detach(), 'scatter': dm1.to(device)},
        params.dx,
        pml_width=[0, 20, 20, 20, 0, 0]
    ).to(device)
    
    running_loss = 0
    num_batches = params.num_batches
    
    optimizer.zero_grad()
    
    for it in range(num_batches):  # loop over shots
        batch_wavl = wavelet[:, it::num_batches].to(device)
        batch_x_s = params.s_cor[it::num_batches].to(device)
        batch_x_r = params.r_cor[it::num_batches].to(device)
        
        batch_data_pred = prop(batch_wavl, batch_x_s, batch_x_r, params.dt)
        
        loss = criterion(batch_data_pred, torch.zeros_like(batch_data_pred))
        running_loss += loss.item()
        
        loss.backward()
    
    gc.collect()
    torch.cuda.empty_cache()
    
    return dm1.grad.clone()


# def create_scattering_model(start_location, end_location, spacing, size):
#     psf_scatt = np.zeros(size)
#     psfin = start_location
#     psfend = end_location
#     psfj = spacing
#     # psfsize = (31, 31)
#     psf_scatt[psfin[0]:psfend[0]:psfj[0], psfin[1]:psfend[-1]:psfj[-1]] = 1

#     # PSF grid
#     psfz = np.arange(psfin[0], nz+psfend[0], psfj[0])
#     psfx = np.arange(psfin[1], nx+psfend[1], psfj[1])
#     Psfx, Psfz = np.meshgrid(psfx, psfz, indexing='ij')
    
#     return psf_scatt, (Psfx, Psfz)


def create_scattering_model(start_location: tuple[int, int], end_location: tuple[int, int], spacing: tuple[int, int],
                            size: tuple[int, int]) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Create a scattering model represented by a binary array.

    Parameters:
    - start_location (tuple[int, int]): Starting location (row, column) of the scattering region.
    - end_location (tuple[int, int]): Ending location (row, column) of the scattering region.
    - spacing (tuple[int, int]): Spacing (row, column) between scattered points in the model.
    - size (tuple[int, int]): Size (height, width) of the model array.

    Returns:
    - psf_scatt (np.ndarray): Scattering model represented by a binary array.
    - (Psfx, Psfz) (tuple[np.ndarray, np.ndarray]): Grid coordinates (x, z) of the scattering model.

    """
    psf_scatt = np.zeros(size)
    psfin = start_location
    psfend = end_location
    psfj = spacing

    psf_scatt[psfin[0]:psfend[0]:psfj[0], psfin[1]:psfend[-1]:psfj[-1]] = 1

    psfz = np.arange(psfin[0], size[0] + psfend[0], psfj[0])
    psfx = np.arange(psfin[1], size[1] + psfend[1], psfj[1])
    Psfx, Psfz = np.meshgrid(psfx, psfz, indexing='ij')

    return psf_scatt, (Psfx, Psfz)


# def create_psfs(scattering_model, psfs_locations_z, psfs_locations_x, psfs_location_init, psfsize):
#     psfs = np.zeros((len(psfs_locations_z), len(psfs_locations_x), *psfsize))
#     for ipz, pz in enumerate(psfs_locations_z):
#         for ipx, px in enumerate(psfs_locations_x):
#             psfs[ipz, ipx] = \
#             scattering_model[psfs_location_init[0]+int(ipz*psfsize[0]):int(ipz*psfsize[0])+psfs_location_init[0]+psfsize[0],
#             psfs_location_init[1]+int(ipx*psfsize[1]):int(ipx*psfsize[1])+psfs_location_init[1]+psfsize[1]]
#     return psfs

def create_psfs(scattering_model: np.ndarray, psfs_locations_z: list[int], psfs_locations_x: list[int],
                psfs_location_init: tuple[int, int], psfsize: tuple[int, int]) -> np.ndarray:
    """Create Point Spread Functions (PSFs) from a scattering model at specified locations.

    Parameters:
    - scattering_model (np.ndarray): Scattering model represented by a binary array.
    - psfs_locations_z (list[int]): List of z-coordinates of the PSF locations.
    - psfs_locations_x (list[int]): List of x-coordinates of the PSF locations.
    - psfs_location_init (tuple[int, int]): Initial location (row, column) of the PSFs in the scattering model.
    - psfsize (tuple[int, int]): Size (height, width) of each PSF.

    Returns:
    - psfs (np.ndarray): Array of Point Spread Functions (PSFs) created from the scattering model.

    """
    psfs = np.zeros((len(psfs_locations_z), len(psfs_locations_x), *psfsize))
    
    for ipz, pz in enumerate(psfs_locations_z):
        for ipx, px in enumerate(psfs_locations_x):
            psfs[ipz, ipx] = scattering_model[
                psfs_location_init[0] + int(ipz * psfsize[0]):int(ipz * psfsize[0]) + psfs_location_init[0] + psfsize[0],
                psfs_location_init[1] + int(ipx * psfsize[1]):int(ipx * psfsize[1]) + psfs_location_init[1] + psfsize[1]
            ]
    
    return psfs



# def run_inversion(params,prop, prop_born, data_t,wavelet,msk,niter, gmax, dm1max, itr=None, device=None,**kwargs): 
#    """ 
#    This run the FWI inversion,  
#    ===================================
#    Arguments: 
#       model: torch.Tensor [nz.nx]: 
#          Initial model for FWI 
#       data_t: torch.Tensor [nt,ns,nr]: 
#          Observed data
#       wavelet: torch.Tensor [nt,1,1] or [nt,ns,1]
#          wavelet 
#       msk: torch.Tensor [nz,nx]:
#          Mask for water layer
#       niter: int: 
#          Number of iteration 
#       device: gpu or cpu  
#       ==================================
#    Optional: 
#       vmin: int:
#          upper bound for the update 
#       vmax: int: 
#          lower bound for the update 
#       smth_flag: bool: 
#          smoothin the gradient flag 
#       smth: sequence of tuble or list: 
#          each element define the amount of smoothing for different axes
#       plot_flag: bool 
#             Excute plotting command for gradients and model updates
#       method: string ("1D" or None)     
#             Apply the gradient stacking for 1D models    
#       """



#    wavelet = wavelet.to(device).contiguous()
#    msk = msk.int().to(device)
#    #    num_shots_per_batch = int(params.num_shots / num_batches)
#    # prop = params.propagator(device)
#    #    t_start = time.time()
#    loss_iter=[]

#    # updates is the output file
#    updates = []
#    gradients = []
#    dm1s = []    

#    d = torch.tensor(list(range(params.nz))*params.nx).reshape(params.nx, params.nz).T /params.nz
#    d = d.to(device)
#    for itr in range(niter):
#          running_loss = 0 
#          params.optimizer.zero_grad()
#          for it in range(num_batches): # loop over shots 
#             batch_wavl = wavelet[:,it::num_batches]
#             batch_data_t = data_t[:,it::num_batches].to(device)
#             batch_x_s = params.s_cor[it::num_batches].to(device)
#             batch_x_r = params.r_cor[it::num_batches].to(device)
#             batch_data_pred = prop(batch_wavl, batch_x_s, batch_x_r, params.dt)
#             loss = criterion(batch_data_pred, batch_data_t)
#             #  if loss.item() == 0.0: 
#             #     updates.append(model.detach().cpu().numpy())
#             #     return np.array(updates)
#             loss.backward()            
#             running_loss += loss.item()   

#       #   if smth_flag: model.grad = params.grad_smooth(model,smth[1],smth[0]).to(device)            
#          if itr == 0: gmax = (torch.abs(params.model.grad.detach())).max() # get max of first itr 
#          params.model.grad = (params.model.grad / gmax) * msk   # normalize 

#          gradients.append(params.model.grad.detach().clone())

#          params.dm1.data = params.model.grad.detach().clone()
#          params.dm1.requires_grad = True

#          torch.cuda.empty_cache()
#          gc.collect()

#          running_lossb = 0 
#          params.optimizer_dm1.zero_grad()
#          for it in range(num_batches): # loop over shots 
#             batch_wavl = wavelet[:,it::num_batches]
#             batch_x_s = params.s_cor[it::num_batches].to(device)
#             batch_x_r = params.r_cor[it::num_batches].to(device)
#             batch_data_pred = prop_born(batch_wavl, batch_x_s, batch_x_r, params.dt)
#             lossb = criterion(batch_data_pred, torch.zeros_like(batch_data_pred))
#             lossb.backward()            
#             running_lossb += lossb.item()

#          if itr == 0: dm1max =  1e1 * torch.abs(params.dm1.grad.detach()).max()
#          params.dm1.grad = (params.dm1.grad / dm1max) * msk
#          dm1s.append(params.dm1.grad.detach().clone())

#       #   x = dm1s[-1]
#       #   y = gradients[-1]
#       #   train_dataloader = prepare_data(x, d, y, patch_size=32, slide=8, batch_size=64)

#       #   optimizer.step()

#       #   model.data[model.data < m_min] = m_min
#       #   model.data[model.data > m_max] = m_max
#          loss_iter.append(running_loss)

#          torch.cuda.empty_cache()
#          gc.collect()
#       #   print('Iteration: ', itr, 'Objective: ', running_loss) 

#       #   updates.append(model.detach().clone().cpu())  

#       #   if plot_flag and itr%1==0:
#       #        plt.figure(figsize=(12,8))
#       #        dm_vmin, dm_vmax = np.percentile(model.grad.detach().cpu(), [2,98])
#       #        plt.imshow(model.grad.detach().cpu(),vmin=dm_vmin, vmax=dm_vmax, cmap='terrain')
#       #        plt.title('Gradient', fontsize=18)
#       #        plt.axis('tight')
#       #        plt.colorbar(shrink=0.62, pad=0.02)
#       #        plt.show()



#       #        if itr>1:   
#       #             plt.figure(figsize=(12,8))
#       #             dm_vmin, dm_vmax = np.percentile(updates[itr], [2,98])
#       #             plt.imshow(updates[itr],vmin=dm_vmin, vmax=dm_vmax, cmap='terrain')
#       #             plt.title('Updated Model', fontsize=18)
#       #             plt.axis('tight')
#       #             plt.colorbar(shrink=0.62, pad=0.02)
#       #             plt.show()   

# #    # End of FWI iteration
# #    t_end = time.time()
# #    print('Runtime in min :',(t_end-t_start)/60)
#    if itr == 0:
#       return loss_iter, gradients, dm1s, gmax, dm1max       
#    return loss_iter, gradients, dm1s