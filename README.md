# CroMoTEX: Contrastive Cross-Modal Learning for Infusing Chest X-ray Knowledge into ECGs


## Dataset download
Refer `data_download_steps.md`

## Environment set up
1. `conda create --prefix ./venv python=3.12`
1. `conda activate ./venv`
1. `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
1. `pip install tqdm pandas imageio opencv-python h5py wfdb hydra-core mlflow rich torchxrayvision`
1. `pip install -e .`

## Data pre-processing
1. `python preprocess/unzip_datasets.py`
1. `python preprocess/prepare_dataset.py`

## Training
1. Pre-train CXR encoder: `python scripts/pretrain_img_classif.py`
2. Pre-train ECG encoder: `python scripts/pretrain_ecg_encoder.py`
3. Multimodal training: `python scripts/train_multimodal.py`
4. Fine-tune: `python scripts/finetune.py`

## Notes
1. All config options/hyperparams are in `config/config.yaml` and `config/cromotex/cromotex_patch_transformer.yaml`.
