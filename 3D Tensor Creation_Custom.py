#!/usr/bin/env python
# coding: utf-8

import os
import torch
import time
import pandas as pd
import nibabel as nib
import numpy as np
import random
import re
import glob
from scipy.ndimage import zoom
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm

import gc
gc.collect()
torch.cuda.empty_cache()

# Configuration
DATA_CSV_PATH = '/workspace/pumpkinlab-storage-dhl/tabular/ADNI_master_merged_12-17-2025.csv'
OUTPUT_PATH = '/workspace/BrainMRI-3way-classification/data'
MRI_BASE_PATH = '/workspace/pumpkinlab-storage-dhl'
MRI_FOLDERS = [
    'all_brain_mni152_1mm_02_04_2026'
]

config = {
    'img_size': 192,
    'depth': 192
}

random.seed(37)


class DataPaths():
    def __init__(self, csv_path=None):
        if csv_path is None:
            self.csv_path = DATA_CSV_PATH
        else:
            self.csv_path = csv_path

    def _collect_all_mri_files(self):
        """Collect all MRI files from ADNI_Download folders"""
        print("Collecting MRI files from all folders...")
        all_files = []

        for folder in MRI_FOLDERS:
            folder_path = os.path.join(MRI_BASE_PATH, folder)
            if os.path.exists(folder_path):
                files = glob.glob(os.path.join(folder_path, '*.nii.gz'))
                all_files.extend(files)
                print(f"  {folder}: {len(files)} files")

        print(f"Total MRI files found: {len(all_files)}")

        # Parse filenames to extract S#####_I###### pattern
        file_info = {}
        for filepath in all_files:
            filename = os.path.basename(filepath)

            # Extract S#####_I###### pattern using regex
            pattern = r'(S\d+)_(I\d+)'
            match = re.search(pattern, filename)

            if match:
                s_id = match.group(1)
                i_id = match.group(2)
                si_pattern = f"{s_id}_{i_id}"
                file_info[si_pattern] = filepath

        print(f"Parsed {len(file_info)} unique S#####_I###### patterns")
        return file_info

    def patient_id_loading(self):
        # Collect all MRI files
        file_info = self._collect_all_mri_files()

        # Load CSV
        df = pd.read_csv(self.csv_path, low_memory=False)
        print(f"\nTotal sessions in CSV: {len(df)}")

        cn_mri_scan_list, mci_mri_scan_list, ad_mri_scan_list = [], [], []
        matched_count = 0
        unmatched_count = 0

        for idx, row in df.iterrows():
            # Skip if DX is NaN
            if pd.isna(row['DX']):
                continue

            # Extract S#####_I###### pattern from New_Path
            new_path = row['New_Path']
            pattern = r'(S\d+)_(I\d+)'
            match = re.search(pattern, new_path)

            if match:
                s_id = match.group(1)
                i_id = match.group(2)
                si_pattern = f"{s_id}_{i_id}"

                # Find matching file
                if si_pattern in file_info:
                    image_dict = {}
                    image_dict['image_path'] = file_info[si_pattern]
                    image_dict['patient_id'] = row['Subject']
                    image_dict['image_id'] = str(row['Image Data ID'])

                    # Map Dementia to AD
                    label = row['DX']
                    if label == 'Dementia':
                        label = 'AD'
                    image_dict['label'] = label

                    if label == 'CN':
                        cn_mri_scan_list.append(image_dict)
                    elif label == 'MCI':
                        mci_mri_scan_list.append(image_dict)
                    elif label == 'AD':
                        ad_mri_scan_list.append(image_dict)

                    matched_count += 1
                else:
                    unmatched_count += 1

        print(f"\nMatching results:")
        print(f"  Matched: {matched_count}")
        print(f"  Unmatched: {unmatched_count}")
        print(f"\nClass distribution:")
        print(f"  CN: {len(cn_mri_scan_list)}, MCI: {len(mci_mri_scan_list)}, AD: {len(ad_mri_scan_list)}")

        all_images = cn_mri_scan_list + mci_mri_scan_list + ad_mri_scan_list

        # Create output directory
        save_path = os.path.join(OUTPUT_PATH, 'csv_splits')
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        all_img_df = pd.DataFrame(all_images)
        all_img_df.insert(0, 'pt_index', range(len(all_img_df)))
        all_img_df_path = os.path.join(save_path, 'all_mri_scan_list.csv')
        all_img_df.to_csv(all_img_df_path, index=False)

        return all_img_df_path


class ADNIAlzheimerDataset(Dataset):
    def __init__(self, image_df_paths, transform=None):
        self.image_df_paths = image_df_paths
        self.transform = transform
        self.df = pd.read_csv(self.image_df_paths)
        self.desired_width = config['img_size']
        self.desired_height = config['img_size']
        self.desired_depth = config['depth']
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(15),
            transforms.ToTensor()
        ])

    def __label_extract(self, group):
        if group == 'CN':
            return 0
        elif group == 'MCI':
            return 1
        elif group == 'AD':
            return 2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = {}
        image_filepath = self.df['image_path'][idx]

        # Check if file exists
        if not os.path.exists(image_filepath):
            print(f"Warning: File not found: {image_filepath}")
            # Return empty tensor with correct shape
            return torch.zeros(1, self.desired_width, self.desired_height, self.desired_depth, dtype=torch.float32), -1

        image = nib.as_closest_canonical(nib.load(image_filepath))
        image = image.get_fdata()

        xdim, ydim, zdim = image.shape

        # Pad to 256x256x256
        image = np.pad(
            image,
            [((256 - xdim) // 2, (256 - xdim) // 2),
             ((256 - ydim) // 2, (256 - ydim) // 2),
             ((256 - zdim) // 2, (256 - zdim) // 2)],
            'constant',
            constant_values=0
        )

        # Resize to desired dimensions
        width_factor = self.desired_width / image.shape[0]
        height_factor = self.desired_height / image.shape[1]
        depth_factor = self.desired_depth / image.shape[-1]

        image = zoom(image, (width_factor, height_factor, depth_factor), order=1)
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
        image = image.astype('float32')
        image = torch.from_numpy(image)

        label = self.df['label'][idx]
        label = self.__label_extract(label)

        return image, label


def saveTensors(dataset, delete_original=False):
    data_path = os.path.join(OUTPUT_PATH, '3D_tensors')
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    labels = {
        0: 'CN',
        1: 'MCI',
        2: 'AD'
    }

    # Create label subdirectories
    for label in labels.keys():
        label_path = os.path.join(data_path, labels[label])
        if not os.path.exists(label_path):
            os.makedirs(label_path)

    print(f"Processing is starting. Data will be saved at {data_path}")
    print(f"Total number of images are: {len(dataset)}")

    start = time.time()
    skipped = 0
    deleted = 0

    for idx in tqdm(range(len(dataset)), desc="Saving tensors", unit="img"):
        tensor, label = dataset.__getitem__(idx)

        # Skip if file was not found
        if label == -1:
            skipped += 1
            continue

        tensor_path = f"{data_path}/{labels[label]}/{idx}.pt"
        torch.save(tensor, tensor_path)

        # Delete original file after successful save
        if delete_original:
            original_path = dataset.df['image_path'][idx]
            if os.path.exists(original_path):
                os.remove(original_path)
                deleted += 1

    req_time = time.time() - start
    print(f"Total time required for processing the data is {req_time // 60} minutes {req_time % 60} sec.")
    print(f"Processing of a single image took {req_time / (1.0 * len(dataset))} sec.")

    if skipped > 0:
        print(f"Warning: {skipped} images were skipped due to file not found errors.")
    if deleted > 0:
        print(f"Deleted {deleted} original files to free disk space.")


if __name__ == "__main__":
    print("="*60)
    print("Starting 3D Tensor Creation for Custom ADNI Data")
    print("="*60)

    # Create data paths
    dataPath = DataPaths()
    all_img_df_path = dataPath.patient_id_loading()

    print("\n" + "="*60)
    print("Creating dataset...")
    print("="*60)

    dataset = ADNIAlzheimerDataset(all_img_df_path)

    print("\n" + "="*60)
    print("Saving tensors...")
    print("="*60)

    saveTensors(dataset, delete_original=True)

    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)
    print(f"Output saved to: {OUTPUT_PATH}")
