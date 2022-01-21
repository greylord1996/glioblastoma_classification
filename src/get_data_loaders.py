from src.torch_dataset import Dataset
import torch
from torch.utils import data as torch_data


def prepare_train_valid_dataloader(
        df, fold, num_images, img_size, data_directory, mri_type,
        train_batch_size, valid_batch_size,
        num_workers
):
    df.loc[:, "MRI_Type"] = mri_type
    df_train_this = df[df['fold'] != fold]
    df_valid_this = df[df['fold'] == fold]

    dataset_train = Dataset(
        df_train_this["BraTS21ID"].values,
        num_images,
        img_size,
        data_directory,
        targets=df_train_this["MGMT_value"].values,
        mri_type=df_train_this["MRI_Type"].values,
        split="train"
    )
    dataset_valid = Dataset(
        df_valid_this["BraTS21ID"].values,
        num_images,
        img_size,
        data_directory,
        targets=df_valid_this["MGMT_value"].values,
        mri_type=df_valid_this["MRI_Type"].values,
        split="train"
    )

    train_loader = torch_data.DataLoader(
        dataset_train, batch_size=train_batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True
    )
    valid_loader = torch_data.DataLoader(
        dataset_valid, batch_size=valid_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, valid_loader
