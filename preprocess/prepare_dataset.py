import pandas as pd
import numpy as np
import ast
import re
import random
import h5py
import contextlib
from rich.logging import RichHandler
from rich.progress import track
import logging
import os
from PIL import Image
import cv2
from torchxrayvision.datasets import XRayCenterCrop, XRayResizer
import wfdb
import wfdb.processing as wp

import cromotex.utils.preprocess as preprocess

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(show_level=False, markup=True, show_path=False)]
)

logging.info("Reading and processing MIMIC-IV datasets...")

pathologies = [
    "cardiomegaly", "edema",
    "enlarged_cardiomediastinum", "pleural_effusion", "pneumonia"
]

def read_and_process_datasets():
    """
    Read MIMIC-CXR-JPG, MIMIC-IV-ECG, MIMIC-IV-ED, MIMIC-IV-Note datasets
    Clean all these datasets and merge them into a single dataframe
    Perform train-val-test splits
    Save the train-val-test dataframes as pickle files
    """
    ### CXR data
    df_cxr_labels = pd.read_csv(
        "datasets/mimic-cxr-jpg/mimic-cxr-2.0.0-chexpert.csv"
    )

    pathology_str = [
        "Cardiomegaly", "Edema", "Enlarged Cardiomediastinum",
        "Pleural Effusion", "Pneumonia"
    ]
    cols = ["subject_id", "study_id"] + pathology_str
    df_cxr_labels = df_cxr_labels[cols]
    df_cxr_labels = df_cxr_labels.fillna(-2)
    df_cxr_labels = df_cxr_labels.replace(-1, 0)
    df_cxr_labels = df_cxr_labels.replace(-2, 0)

    df_cxr_meta = pd.read_csv(
        "datasets/mimic-cxr-jpg/mimic-cxr-2.0.0-metadata.csv"
    )
    df_cxr_meta = df_cxr_meta[df_cxr_meta["ViewPosition"].isin(["AP", "PA"])]

    df_cxr = df_cxr_labels.merge(
        df_cxr_meta, on=["subject_id", "study_id"], how="inner"
    )

    df_cxr['StudyDate'] = df_cxr['StudyDate'].astype(str)
    df_cxr['StudyTime'] = df_cxr['StudyTime'].astype(str)
    df_cxr['StudyTime'] = df_cxr['StudyTime'].apply(lambda x: x.split('.')[0])
    df_cxr['StudyTime'] = df_cxr['StudyTime'].apply(lambda x: x.zfill(6))
    df_cxr['cxr_date'] = df_cxr['StudyDate'].apply(
        lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:]
    )
    df_cxr['cxr_time'] = df_cxr['StudyTime'].apply(
        lambda x: x[:2] + ':' + x[2:4] + ':' + x[4:]
    )

    df_cxr['cxr_timestamp'] = pd.to_datetime(
        df_cxr['cxr_date'] + ' ' + df_cxr['cxr_time'],
        format='%Y-%m-%d %H:%M:%S'
    )

    df_cxr = df_cxr.rename(
        columns={
            "ViewPosition": "view_position",
            "Cardiomegaly": "cardiomegaly",
            "Edema": "edema",
            "Enlarged Cardiomediastinum": "enlarged_cardiomediastinum",
            "Pleural Effusion": "pleural_effusion",
            "Pneumonia": "pneumonia",
            "study_id": "cxr_study_id"
        }
    )

    df_cxr = df_cxr[
        [
            "subject_id", "cxr_study_id", "dicom_id", "view_position",
            "cxr_timestamp", "cardiomegaly", "edema", 
            "enlarged_cardiomediastinum", "pleural_effusion", "pneumonia"
        ]
    ]

    ### ECG data
    df_ecg_icd10 = pd.read_csv(
        'datasets/mimic-iv-ecg-ext-icd-labels/1.0.1/records_w_diag_icd10.csv'
    )
    df_ecg_icd10 = df_ecg_icd10[df_ecg_icd10['all_diag_all'] != "[]"]
    df_ecg_icd10['all_diag_all'] = df_ecg_icd10['all_diag_all'].apply(
        ast.literal_eval
    )

    df_ed = pd.read_csv('datasets/mimic-iv-ed/2.2/ed/edstays.csv')

    df_ecg = pd.merge(
        df_ecg_icd10, df_ed, left_on=['subject_id', 'ed_stay_id'],
        right_on=['subject_id', 'stay_id'], how='left'
    )

    df_ecg['intime'] = pd.to_datetime(
        df_ecg['intime'], format='%Y-%m-%d %H:%M:%S'
    )
    df_ecg['outtime'] = pd.to_datetime(
        df_ecg['outtime'], format='%Y-%m-%d %H:%M:%S'
    )

    df_ecg = df_ecg[df_ecg['intime'].notnull()]
    df_ecg = df_ecg[df_ecg['outtime'].notnull()]

    df_ecg = df_ecg.rename(columns={'study_id': 'ecg_study_id'})
    keep_cols = [
        'subject_id', 'ecg_study_id', 'file_name', 'intime',
        'outtime', 'ed_hadm_id', 'hosp_hadm_id', 'ed_stay_id'
    ]

    df_ecg = df_ecg[keep_cols]
    df_ecg['combined_hadm_id'] = df_ecg['ed_hadm_id'].fillna(
        df_ecg['hosp_hadm_id']
    )

    ### Match ECG with CXR
    logging.info("Matching ECG with chest x-ray data...")

    def match_cxr_with_ecg(df_cxr, df_ecg):
        df_ecg['matched'] = False

        matched_filenames = []
        matched_study_ids = []
        matched_intimes = []
        matched_outtimes = []
        matched_combined_hadm_ids = []
        matched_ed_stay_ids = []

        for index, cxr_row in track(
            df_cxr.iterrows(), total=len(df_cxr),
            description="Matching CXR to ECGs"
        ):
            matching_ecgs = df_ecg[
                (df_ecg['subject_id'] == cxr_row['subject_id']) &
                (df_ecg['intime'] <= cxr_row['cxr_timestamp']) &
                (df_ecg['outtime'] >= cxr_row['cxr_timestamp'])
            ]

            # Check if any matching ECGs are unmatched
            unmatched_ecgs = matching_ecgs[~matching_ecgs['matched']]

            if not unmatched_ecgs.empty:
                # Pick the first unmatched ECG and mark it as matched
                chosen_ecg = unmatched_ecgs.iloc[0]
                df_ecg.loc[chosen_ecg.name, 'matched'] = True
            elif not matching_ecgs.empty:
                # If all matching ECGs are already matched, pick the first one
                chosen_ecg = matching_ecgs.iloc[0]
            else:
                # No matching ECGs found
                chosen_ecg = None

            # Append matched ECG details or None for unmatched
            matched_filenames.append(
                chosen_ecg['file_name'] if chosen_ecg is not None else None
            )
            matched_study_ids.append(
                chosen_ecg['ecg_study_id'] if chosen_ecg is not None else None
            )
            matched_intimes.append(
                chosen_ecg['intime'] if chosen_ecg is not None else None
            )
            matched_outtimes.append(
                chosen_ecg['outtime'] if chosen_ecg is not None else None
            )
            matched_combined_hadm_ids.append(
                chosen_ecg['combined_hadm_id']
                if chosen_ecg is not None
                else None
            )
            matched_ed_stay_ids.append(
                chosen_ecg['ed_stay_id'] if chosen_ecg is not None else None
            )
        # Add the matched ECG details as new columns in df_cxr
        df_cxr['ecg_filename'] = matched_filenames
        df_cxr['ecg_study_id'] = matched_study_ids
        df_cxr['ecg_intime'] = matched_intimes
        df_cxr['ecg_outtime'] = matched_outtimes
        df_cxr['ecg_combined_hadm_id'] = matched_combined_hadm_ids
        df_cxr['ecg_ed_stay_id'] = matched_ed_stay_ids

        return df_cxr

    df_merged = match_cxr_with_ecg(df_cxr, df_ecg)

    ### Notes dataset
    df_notes = pd.read_csv('datasets/mimic-iv-note/2.2/note/discharge.csv')

    def extract_chief_complaint_section(discharge_text):
        # Use regex to find text between "Chief complaint"
        # and "Physical exam", case-insensitive
        match = re.search(
            r'chief complaint.*?physical exam',
            discharge_text, flags=re.IGNORECASE | re.DOTALL
        )
        if match:
            return match.group(0)
        return None 

    df_notes['text_clipped'] = df_notes['text'].apply(
        extract_chief_complaint_section
    )
    df_notes['text_clipped'] = df_notes['text_clipped'].str.replace('\n', ' ')

    df_merged = pd.merge(
        df_merged,
        df_notes,
        left_on=['subject_id', 'ecg_combined_hadm_id'],
        right_on=['subject_id', 'hadm_id'],
        how='left'
    )

    df_triage = pd.read_csv('datasets/mimic-iv-ed/2.2/ed/triage.csv')

    df_triage['chiefcomplaint'] = (
        'Chief Complaint: ' + df_triage['chiefcomplaint'].astype(str)
    )

    df_merged = pd.merge(
        df_merged,
        df_triage,
        left_on=['subject_id', 'ecg_ed_stay_id'],
        right_on=['subject_id', 'stay_id'],
        how = 'left'
    )

    df_merged['clinical_text'] = df_merged['text_clipped'].fillna(
        df_merged['chiefcomplaint']
    )

    pfx = "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/files/"
    df_merged['ecg_filename'] = df_merged['ecg_filename'].apply(
        lambda x: x.replace(
            pfx,
            ""
        ) if x is not None else None
    )

    def construct_cxr_filename(row):
        if pd.notna(row['cxr_study_id']) and pd.notna(row['dicom_id']):
            # Construct filename if cxr_study_id and cxr_dicom_id are available
            return (
                "p" + str(int(row['subject_id']))[:2] + "/p"
                + str(int(row['subject_id'])) + "/s"
                + str(int(row['cxr_study_id'])) + "/"
                + row['dicom_id'] + ".jpg"
            )
        else:
            return None  # Return None if necessary fields are missing

    df_merged['cxr_filename'] = df_merged.apply(construct_cxr_filename, axis=1)

    if not os.path.exists('datasets/processed_csv'):
        os.makedirs('datasets/processed_csv')

    df_merged.to_csv(
        "datasets/processed_csv/df_merged_all_cols.csv", index=False
    )

    keep_cols = [
        "subject_id",
        "ecg_filename",
        "cxr_filename",
        "clinical_text",
        "cardiomegaly",
        "edema",
        "enlarged_cardiomediastinum",
        "pleural_effusion",
        "pneumonia",
    ]

    df_merged = df_merged[keep_cols]
    df_merged.to_csv("datasets/processed_csv/df_merged.csv", index=False)

    logging.info("Saved full dataset to datasets/processed_csv/df_merged.csv")

    ### Train-val-test split
    unique_subject_ids = df_merged['subject_id'].unique().tolist()

    unique_subject_ids.sort() #for reproducibility

    random_seed = 42 #for reproducibility
    random.seed(random_seed)
    random.shuffle(unique_subject_ids)

    train_size = 0.7
    val_size = 0.1
    test_size = 0.2

    train_subject_ids = unique_subject_ids[
        :int(len(unique_subject_ids) * train_size)
    ]
    val_subject_ids = unique_subject_ids[
        int(
            len(unique_subject_ids)
            * train_size):int(len(unique_subject_ids)
            * (train_size + val_size)
        )
    ]
    test_subject_ids = unique_subject_ids[
        int(len(unique_subject_ids) * (train_size + val_size)):
    ]

    df_train = df_merged[df_merged['subject_id'].isin(train_subject_ids)]
    df_val = df_merged[df_merged['subject_id'].isin(val_subject_ids)]
    df_test = df_merged[df_merged['subject_id'].isin(test_subject_ids)]

    logging.info(
        f"train/val ratio: {len(df_train)/len(df_val):.2f}"
        f" | expected: 7.00"
    )

    logging.info(
        f"train/test ratio: {len(df_train)/len(df_test):.2f}"
        f" | expected: 3.50"
    )

    if not os.path.exists('datasets/processed'):
        os.makedirs('datasets/processed')
        
    df_train.to_pickle('datasets/processed/df_train.pkl')
    df_val.to_pickle('datasets/processed/df_val.pkl')
    df_test.to_pickle('datasets/processed/df_test.pkl')

    logging.info(
        "Saved train, val and test pickle files to to datasets/processed"
    )

def create_cxr_pretrain_dataset():
    """
    Use the train df from above to create a pretrain dataset for
    the CXR encoder. Split this into pretrain_train and pretrain_val.
    Save the dataset into a hdf5 file.
    """
    ### Create HDF5 files for pre-training the CXR encoder

    df_train = pd.read_pickle('datasets/processed/df_train.pkl')

    df_pretrain = df_train[df_train['cxr_filename'].notna()]
    df_pretrain = df_pretrain.reset_index(drop=True)

    df_pretrain_train = df_pretrain.head(int(len(df_pretrain) * 0.8))
    df_pretrain_train = df_pretrain_train.reset_index(drop=True)

    df_pretrain_val = df_pretrain.tail(int(len(df_pretrain) * 0.2))
    df_pretrain_val = df_pretrain_val.reset_index(drop=True)

    def create_hdf5_from_images(df, hdf5_filename, size=224):
        """
        Center-crop chest x-ray images, resize to size x size and save 
        them in an HDF5 file.
        """

        if not os.path.exists('datasets/processed'):
            os.makedirs('datasets/processed')
        
        hdf5_file_path = f'datasets/processed/{hdf5_filename}.h5'

        label_columns = pathologies

        with h5py.File(hdf5_file_path, 'w') as hdf5_file:
            images_dataset = hdf5_file.create_dataset(
                'images',
                shape=(len(df), 3, size, size),
                dtype=np.float32
            )

            labels_dataset = hdf5_file.create_dataset(
                'labels',
                shape=(len(df), len(label_columns)),
                dtype=np.float32
            )
            
            center_crop = XRayCenterCrop()
            
            # suppress meaningless print statements 
            # by xrv.datasets.XRayResizer
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    resizer = XRayResizer(size, engine='cv2')

            for idx, row in track(
                df.iterrows(), total=len(df),
                description="Processing CXR"
            ):
                file_path = (
                    "datasets/mimic-cxr-jpg/files/" + row["cxr_filename"]
                )
                
                #Convert to grayscale ('L')
                img = Image.open(file_path).convert('L')
                img = np.array(img)
                
                img = img / 255.0
                img = np.stack([img] * 3, axis=0)

                img = center_crop(img)
                img = resizer(img)
                        
                # Store in the dataset
                images_dataset[idx] = img
                labels_dataset[idx] = row[label_columns].values.astype(
                    np.float32
                )
                
        logging.info(f"HDF5 file saved to {hdf5_file_path}")

    create_hdf5_from_images(df_pretrain_train, 'pretrain_train', size=224)
    create_hdf5_from_images(df_pretrain_val, 'pretrain_val', size=224)

def create_ecg_pretrain_dataset():
    """
    Create ECG pretrain dataset.
    Since this will be used for uni-modal contrastive learning of ECG 
    without needing label information, this will include all ECGs 
    present in MIMIC-IV-ECG except those with subject_id in the test set.
    Save the datasets into a hdf5 file.
    """

    logging.info("Creating HDF5 files for ECG pretrain data...")

    df_ecg_icd10 = pd.read_csv(
        'datasets/mimic-iv-ecg-ext-icd-labels/1.0.1/records_w_diag_icd10.csv'
    )
    
    df_test = pd.read_pickle('datasets/processed/df_test.pkl')

    test_subject_ids = df_test['subject_id'].unique().tolist()

    df_ecg_pretrain = df_ecg_icd10[
        ~df_ecg_icd10['subject_id'].isin(test_subject_ids)
    ]

    pretain_subject_ids = df_ecg_pretrain['subject_id'].unique().tolist()

    pretain_subject_ids.sort() #for reproducibility

    random_seed = 42 #for reproducibility
    random.seed(random_seed)
    random.shuffle(pretain_subject_ids)

    pretrain_train_size = 0.8
    pretrain_val_size = 0.2

    pretrain_train_subject_ids = pretain_subject_ids[
        :int(len(pretain_subject_ids) * pretrain_train_size)
    ]
    pretrain_val_subject_ids = pretain_subject_ids[
        int(len(pretain_subject_ids) * pretrain_train_size):
    ]

    keep_cols = ['file_name', 'subject_id']
    df_ecg_pretrain = df_ecg_pretrain[keep_cols]
    df_ecg_pretrain = df_ecg_pretrain.rename(
        columns={'file_name': 'ecg_filename'}
    )
    pfx = "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/files/"
    df_ecg_pretrain['ecg_filename'] = df_ecg_pretrain['ecg_filename'].apply(
        lambda x: x.replace(
            pfx,
            ""
        ) if x is not None else None
    )

    df_ecg_pretrain = df_ecg_pretrain.reset_index(drop=True)
    logging.info(f"len(df_ecg_pretrain): {len(df_ecg_pretrain)}")

    df_ecg_pretrain['ecg_file_path'] = (
        "datasets/mimic-iv-ecg/1.0/files/"
        + df_ecg_pretrain['ecg_filename']
        + ".hea"
    )
    df_ecg_pretrain = df_ecg_pretrain[
        df_ecg_pretrain['ecg_file_path'].apply(os.path.exists)
    ].reset_index(drop=True)

    df_ecg_pretrain.drop(columns=['ecg_file_path'], inplace=True)

    logging.info(f"len(df_ecg_pretrain): {len(df_ecg_pretrain)}")

    df_ecg_pretrain_train = df_ecg_pretrain[
        df_ecg_pretrain['subject_id'].isin(pretrain_train_subject_ids)
    ].reset_index(drop=True)

    df_ecg_pretrain_val = df_ecg_pretrain[
        df_ecg_pretrain['subject_id'].isin(pretrain_val_subject_ids)
    ].reset_index(drop=True)

    train_val_ratio = len(df_ecg_pretrain_train)/len(df_ecg_pretrain_val)

    logging.info(
        f"pretrain_train/pretrain_val ratio: {train_val_ratio:.2f}"
        f" expected: 4.00"
    )

    df_ecg_pretrain_train.to_csv(
        'datasets/processed_csv/df_ecg_pretrain_train.csv',
        index=False
    )
    df_ecg_pretrain_val.to_csv(
        'datasets/processed_csv/df_ecg_pretrain_val.csv',
        index=False
    )

    # df_ecg_pretrain_train = df_ecg_pretrain_train[df_ecg_pretrain_train['ecg_filename'].str.contains('p12370921')].reset_index(drop=True)

    def create_hdf5_ecg_pretrain(df, hdf5_filename):
        """
        Pre-process ECGs using the following steps:
            1. Replace NaNs with 0.0
            2. Ensure lead order is consistent, else reorder
            3. Downsample from 500Hz to 100Hz
            4. Remove baseline wander
            5. Normalize per-lead with min = -1.0 and max = 1.0
        Ref. Appendix D in https://arxiv.org/pdf/2410.16239
        """

        if not os.path.exists('datasets/processed'):
            os.makedirs('datasets/processed')
        
        hdf5_file_path = f'datasets/processed/{hdf5_filename}.h5'

        with h5py.File(hdf5_file_path, 'w') as hdf5_file:
            ecg_dataset = hdf5_file.create_dataset(
                'ecg',
                shape=(len(df), 12, 1000),
                dtype=np.float32
            )

            for idx, row in track(
                df.iterrows(), total=len(df),
                description="Processing ECG"
            ):  
                # ECG processing
                ecg_file = (
                    "datasets/mimic-iv-ecg/1.0/files/" + row['ecg_filename']
                )
                signal, fields = wfdb.rdsamp(ecg_file)
                signal[np.isnan(signal)] = 0.0
                signal = preprocess.ecg_consistency(signal, fields)
                signal = preprocess.resample_signal_poly(
                    signal, fields['fs'], 100
                )

                signal = preprocess.baseline_wander_removal(
                    signal.T, sampling_frequency=100
                ).T

                signal = preprocess.normalize_per_lead(signal)
                signal = signal.T #shape [12, 1000]

                # Store in the dataset
                ecg_dataset[idx] = signal
                
        logging.info(f"HDF5 file saved to {hdf5_file_path}")

    create_hdf5_ecg_pretrain(df_ecg_pretrain_train, 'pretrain_ecg_train')
    create_hdf5_ecg_pretrain(df_ecg_pretrain_val, 'pretrain_ecg_val')

def create_matched_dataset():
    """
    Create a matched subset of train and val datasets where each sample 
    consists of a CXR image and a matched ECG signal, and the disease labels.
    (Note: train dataset contains samples with missing ECGs)
    Save the dataset into a hdf5 file.
    """

    logging.info("Creating HDF5 files with matched ECG and CXR data...")

    df_train = pd.read_pickle('datasets/processed/df_train.pkl')
    df_val = pd.read_pickle('datasets/processed/df_val.pkl')
    df_test = pd.read_pickle('datasets/processed/df_test.pkl')

    df_train_matched = df_train[df_train['ecg_filename'].notnull()]
    df_val_matched = df_val[df_val['ecg_filename'].notnull()]
    df_test_matched = df_test[df_test['ecg_filename'].notnull()]

    df_train_matched = df_train_matched.reset_index(drop=True)
    df_val_matched = df_val_matched.reset_index(drop=True)
    df_test_matched = df_test_matched.reset_index(drop=True)

    def create_hdf5_ecg_cxr_matched(df, hdf5_filename, size=224):
        """
        Center-crop chest x-ray images, resize to size x size.
        Pre-process ECGs using the following steps:
            1. Replace NaNs with 0.0
            2. Ensure lead order is consistent, else reorder
            3. Downsample from 500Hz to 100Hz
            4. Remove baseline wander
            5. Normalize per-lead with min = -1.0 and max = 1.0
        Ref. Appendix D in https://arxiv.org/pdf/2410.16239
        """

        if not os.path.exists('datasets/processed'):
            os.makedirs('datasets/processed')
        
        hdf5_file_path = f'datasets/processed/{hdf5_filename}.h5'

        label_columns = pathologies

        with h5py.File(hdf5_file_path, 'w') as hdf5_file:
            images_dataset = hdf5_file.create_dataset(
                'images',
                shape=(len(df), 3, size, size),
                dtype=np.float32
            )

            ecg_dataset = hdf5_file.create_dataset(
                'ecg',
                shape=(len(df), 12, 1000),
                dtype=np.float32
            )

            labels_dataset = hdf5_file.create_dataset(
                'labels',
                shape=(len(df), len(label_columns)),
                dtype=np.float32
            )
            
            center_crop = XRayCenterCrop()

            # suppress meaningless print statements 
            # by xrv.datasets.XRayResizer
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    resizer = XRayResizer(size, engine='cv2')

            for idx, row in track(
                df.iterrows(), total=len(df),
                description="Processing CXR and ECG"
            ):  
                # Image processing
                cxr_file = (
                    "datasets/mimic-cxr-jpg/files/" + row["cxr_filename"]
                )
                #Convert to grayscale ('L')
                img = Image.open(cxr_file).convert('L')
                img = np.array(img)
                img = img / 255.0
                img = np.stack([img] * 3, axis=0)
                img = center_crop(img)
                img = resizer(img)
                
                # ECG processing
                ecg_file = (
                    "datasets/mimic-iv-ecg/1.0/files/" + row['ecg_filename']
                )
                signal, fields = wfdb.rdsamp(ecg_file)
                signal[np.isnan(signal)] = 0.0
                signal = preprocess.ecg_consistency(signal, fields)
                signal = preprocess.resample_signal_poly(
                    signal, fields['fs'], 100
                )

                signal = preprocess.baseline_wander_removal(
                    signal.T, sampling_frequency=100
                ).T

                signal = preprocess.normalize_per_lead(signal)
                signal = signal.T #shape [12, 1000]

                # Store in the dataset
                images_dataset[idx] = img
                ecg_dataset[idx] = signal
                labels_dataset[idx] = row[label_columns].values.astype(
                    np.float32
                )
                
        logging.info(f"HDF5 file saved to {hdf5_file_path}")

    create_hdf5_ecg_cxr_matched(df_train_matched, 'train_matched', size=224)
    create_hdf5_ecg_cxr_matched(df_val_matched, 'val_matched', size=224)
    create_hdf5_ecg_cxr_matched(df_test_matched, 'test_matched', size=224)

if __name__ == "__main__":
    read_and_process_datasets()
    create_cxr_pretrain_dataset()
    create_matched_dataset()
    create_ecg_pretrain_dataset()