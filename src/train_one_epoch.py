import numpy as np
import torch
from tqdm import tqdm
# from src.classification.compute_all_metrics import compute_all_metrics
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score


def train_one_epoch(model, device, criterion, optimizer, train_loader, use_amp):
    predictions_list = []
    targets_list = []

    model.train()
    bar = tqdm(train_loader)
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    losses = []
    total_loss = 0
    for step, batch in enumerate(bar, 1):

        images = torch.tensor(batch["X"]).float().to(device)
        print(images.shape)
        targets = batch["y"].to(device)

        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                logits = logits.squeeze(1)
                predictions_list += [F.sigmoid(logits)]
                targets_list += [targets.detach().cpu()]

                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            logits = model(images)
            logits = logits.squeeze(1)

            predictions_list += [F.sigmoid(logits)]
            targets_list += [targets.detach().cpu()]

            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        losses.append(loss.item())

        total_loss = np.mean(losses)

        bar.set_description(f'loss: {loss.item():.5f}, total_loss: {total_loss:.5f}')

    predictions = torch.cat(predictions_list).detach().cpu().numpy()
    targets = torch.cat(targets_list).cpu().numpy()
    metric = roc_auc_score(targets, predictions)

    # loss_train, roc_auc_train, average_precision_train = compute_all_metrics(predictions_list, targets_list,
    #                                                                          losses, target_cols)
    return total_loss, metric
