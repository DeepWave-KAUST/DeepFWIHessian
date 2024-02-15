from typing import Callable, List, Dict
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

def train(
    network: nn.Module,
    training_pair: Dict[str, torch.Tensor],
    optimizer: optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    epochs: int,
    use_scheduler: bool,
    device: torch.device
) -> List[float]:
    """
    Trains a neural network on a given training pair of input-output tensors using the specified loss function and optimizer.

    Parameters
    ----------
    network : nn.Module
        The neural network model to be trained.
    training_pair : Dict[str, torch.Tensor]
        A dictionary containing the input tensor `x` and the output tensor `y` used for training.
        Expected keys are 'x' for inputs and 'y' for target outputs.
    optimizer : optim.Optimizer
        The optimizer used to update the network's weights.
    loss_fn : Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
        The loss function used to calculate the difference between predictions and targets.
    epochs : int
        The number of epochs to train the network.
    use_scheduler : bool
        A flag indicating whether to use a learning rate scheduler or not. If True, a Cosine Annealing learning rate 
        scheduler will be used to adjust the learning rate throughout the epochs.
    device : torch.device
        The device (CPU or GPU) on which to perform the training.

    Returns
    -------
    List[float]
        A list of loss values recorded at the end of each epoch.
    """
    progress_bar = tqdm(total=epochs, desc='Training Progress', leave=False)
    if use_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-6)
    loss_history = []
    network.train()  # Ensure the network is in training mode

    for epoch in range(epochs):
        optimizer.zero_grad()
        grad_pred = network(training_pair['x'].to(device))
        loss_val = loss_fn(grad_pred, training_pair['y'].to(device))
        progress_bar.set_postfix(loss=loss_val.item(), epoch=epoch)
        loss_val.backward()
        optimizer.step()
        scheduler.step()
        loss_history.append(loss_val.item())
        print(f'Training Epoch {epoch}, Loss = {loss_history[-1]}')
        progress_bar.update(1)

    progress_bar.close()
    return loss_history
