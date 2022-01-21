from src.load_images import load_dicom_images_3d
import torch
from torch.utils import data as torch_data


class Dataset(torch_data.Dataset):
    def __init__(self, paths, num_images, img_size, data_directory, targets=None, mri_type=None, split="train"):
        self.paths = paths
        self.targets = targets
        self.mri_type = mri_type
        self.split = split
        self.num_images = num_images,
        self.img_size = img_size,
        self.data_directory = data_directory
        # print("self.num_images = ", self.num_images)
        # print("self.img_size = ", self.img_size)
        # print("self.data_directory = ", self.data_directory)


    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        scan_id = self.paths[index]
        if self.targets is None:
            # print("!!!!!!!!!!!!!!!!!!!!!!", self.num_images)
            data = load_dicom_images_3d(
                str(scan_id).zfill(5),
                num_imgs=self.num_images[0],
                img_size=self.img_size[0],
                data_directory=self.data_directory,
                mri_type=self.mri_type[index],
                split=self.split
            )
        else:
            data = load_dicom_images_3d(
                str(scan_id).zfill(5),
                num_imgs=self.num_images[0],
                img_size=self.img_size[0],
                data_directory=self.data_directory,
                mri_type=self.mri_type[index],
                split="train"
            )

        if self.targets is None:
            return {"X": data, "id": scan_id}
        else:
            return {"X": data, "y": torch.tensor(self.targets[index], dtype=torch.float)}