import os
import shutil
import gzip
import glob
from icecream import ic
from rich.logging import RichHandler
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)

def _unzip_file(gz_file_path, delete_gz=True):
    """
    Unzips a .gz file and saves the unzipped file in the same directory.
    By default deletes the .gz file after unzipping.
    """
    if not os.path.exists(gz_file_path):
        return
    output_path = gz_file_path.rstrip('.gz')
    with gzip.open(gz_file_path, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    if delete_gz:
        os.remove(gz_file_path)

def _remove_gz_files(directory):
    """
    Removes all .gz files in a directory.
    """
    gzfiles = glob.glob(os.path.join(directory, '*.gz'))
    for file in gzfiles:
        try:
            os.remove(file)
        except:
            continue

def prepare_mimic_iv(del_unnecessary_gz=True):
    """
    Unzip patients.csv.gz and diagnoses_icd.csv.gz in MIMIC-IV dataset.
    Delete all other .gz files.
    """
    logging.info("Preparing MIMIC-IV dataset...")
    
    mimic_iv_path = 'datasets/mimiciv'
    version = os.listdir(mimic_iv_path)[0]
    mimic_iv_path = os.path.join(mimic_iv_path, version, 'hosp')
    
    _unzip_file(os.path.join(mimic_iv_path, 'patients.csv.gz'))
    _unzip_file(os.path.join(mimic_iv_path, 'diagnoses_icd.csv.gz'))

    if del_unnecessary_gz:
        _remove_gz_files(mimic_iv_path)


def prepare_mimic_iv_note(del_unnecessary_gz=True):
    """
    Unzip discharge.csv.gz and delete all other .gz files.
    """
    logging.info("Preparing MIMIC-IV-Note dataset...")

    mimic_iv_note_path = 'datasets/mimic-iv-note'
    version = os.listdir(mimic_iv_note_path)[0]
    mimic_iv_note_path = os.path.join(mimic_iv_note_path, version, 'note')

    _unzip_file(os.path.join(mimic_iv_note_path, 'discharge.csv.gz'))

    if del_unnecessary_gz:
        _remove_gz_files(mimic_iv_note_path)

def prepare_mimic_cxr_jpg(del_unnecessary_gz=True):
    """
    Unzip mimic-cxr-2.0.0-metadata.csv.gz and delete all other .gz files.
    """

    logging.info("Preparing MIMIC-CXR-JPG dataset...")

    mimic_cxr_jpg_path = 'datasets/mimic-cxr-jpg'

    _unzip_file(os.path.join(mimic_cxr_jpg_path, 'mimic-cxr-2.0.0-metadata.csv.gz'))
    _unzip_file(os.path.join(mimic_cxr_jpg_path, 'mimic-cxr-2.0.0-split.csv.gz'))
    _unzip_file(os.path.join(mimic_cxr_jpg_path, 'mimic-cxr-2.0.0-chexpert.csv.gz'))
    
    if del_unnecessary_gz:
        _remove_gz_files(mimic_cxr_jpg_path)

def prepare_mimic_iv_ed(del_unnecessary_gz=True):
    """
    Unzip edstays.csv.gx and delete all other .gz files.
    """
    logging.info("Preparing MIMIC-IV-ED dataset...")
    mimic_iv_ed_path = 'datasets/mimic-iv-ed'
    version = os.listdir(mimic_iv_ed_path)[0]
    mimic_iv_ed_path = os.path.join(mimic_iv_ed_path, version, 'ed')

    _unzip_file(os.path.join(mimic_iv_ed_path, 'edstays.csv.gz'))
    _unzip_file(os.path.join(mimic_iv_ed_path, 'triage.csv.gz'))

    if del_unnecessary_gz:
        _remove_gz_files(mimic_iv_ed_path)

def prepare_mimic(del_unnecessary_gz=True):
    """
    Main wrapper function to prepare the entire MIMIC-IV dataset, including
    MIMIC-IV, MIMIC-IV-Note, MIMIC-IV-ECG and MIMIC-CXR-JPG
    """
    prepare_mimic_iv(del_unnecessary_gz)
    prepare_mimic_iv_note(del_unnecessary_gz)
    prepare_mimic_cxr_jpg(del_unnecessary_gz)
    prepare_mimic_iv_ed(del_unnecessary_gz)

if __name__ == '__main__':
    prepare_mimic()