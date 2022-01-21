import torch
import numpy as np
from tqdm import tqdm
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score


def valid_func(model, device, criterion, valid_loader):
    model.eval()
    bar = tqdm(valid_loader)

    targets_list = []
    losses = []
    predictions_list = []

    with torch.no_grad():
        for step, batch in enumerate(bar, 1):
            images = torch.tensor(batch["X"]).float().to(device)
            targets = batch["y"].to(device)

            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            logits = logits.squeeze(1)
            predictions_list += [F.sigmoid(logits)]
            targets_list += [targets.detach().cpu()]
            loss = criterion(logits, targets)
            losses.append(loss.item())
            total_loss = np.mean(losses)
            bar.set_description(f'loss: {loss.item():.5f}, total_loss: {total_loss:.5f}')

        predictions = torch.cat(predictions_list).detach().cpu().numpy()
        targets = torch.cat(targets_list).cpu().numpy()
        metric = roc_auc_score(targets, predictions)

    # loss_valid, roc_auc_valid, average_precision_valid = compute_all_metrics(predictions_list, targets_list, losses,
    #                                                                          target_cols)
    return total_loss, metric