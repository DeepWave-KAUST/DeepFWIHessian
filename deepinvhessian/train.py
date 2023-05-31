
from tqdm.notebook import tqdm

def train(network, dataloader, optimizer, loss_fn, epochs, device):
    loss = []
    network.train()
    for ep in tqdm(range(epochs)):
        running_loss = 0
        for sample in dataloader:
            optimizer.zero_grad()
            x1, y1 = sample[0].to(device), sample[1].unsqueeze(1).to(device)
            dm_pred = network(x1)
            loss_ = loss_fn(dm_pred, y1)
            running_loss += loss_.item()
            loss_.backward()
            optimizer.step()

        loss.append(running_loss/len(dataloader))

        print(f'Training Epoch {ep}, Loss = {loss[-1]}')

        # optimizer_unet.param_groups[-1]['lr'] = lr_init
    return loss