import numpy as np
import pydicom
import cv2
import glob
import re


def load_dicom_image(path, img_size):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    if np.min(data) == np.max(data):
        data = np.zeros((img_size, img_size))
        return data

    data = cv2.resize(data, (img_size, img_size))
    return data


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def load_dicom_images_3d(scan_id, num_imgs, img_size, data_directory, mri_type="FLAIR", split="train"):
    files = natural_sort(glob.glob(f"{data_directory}/{split}/{scan_id}/{mri_type}/*.dcm"))
    # print(num_imgs)
    every_nth = len(files) / num_imgs
    indexes = [min(int(round(i * every_nth)), len(files) - 1) for i in range(0, num_imgs)]

    files_to_load = [files[i] for i in indexes]

    img3d = np.stack([load_dicom_image(f, img_size) for f in files_to_load]).T

    img3d = img3d - np.min(img3d)
    if np.max(img3d) != 0:
        img3d = img3d / np.max(img3d)

    return np.expand_dims(img3d, 0)