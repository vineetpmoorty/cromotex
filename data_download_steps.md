# Steps to download the data


> Access to MIMIC-IV dataset requires credentialing on physionet.org. Refer to [physionet.org](https://physionet.org) for more information, and follow the below steps once you are credentialied.

## MIMIC-CXR-JPG

MIMIC-CXR-JPG is a large dataset with total size of 570.3 GB, and we noticed that downloading directly from physionet.org could take many days. So, we download from the Google Cloud bucket, which we found to be many times faster. Here are the steps we followed:

> **Note**: For this, and all subsequent datasets, consider using `tmux` to obtain a persistent terminal, as downloads could take a long time.

1. Set up [Google Cloud CLI](https://cloud.google.com/sdk/docs/install?authuser=4)
2. `cd <project_root>/datasets`
3. `mkdir mimic-cxr-jpg`
4. `cd mimic-cxr-jpg`
5. Run the below command to download data from the Google Cloud bucket.

```
gsutil -m cp -r \
  "gs://mimic-cxr-jpg-2.1.0.physionet.org/IMAGE_FILENAMES" \
  "gs://mimic-cxr-jpg-2.1.0.physionet.org/LICENSE.txt" \
  "gs://mimic-cxr-jpg-2.1.0.physionet.org/README" \
  "gs://mimic-cxr-jpg-2.1.0.physionet.org/SHA256SUMS.txt" \
  "gs://mimic-cxr-jpg-2.1.0.physionet.org/files" \
  "gs://mimic-cxr-jpg-2.1.0.physionet.org/mimic-cxr-2.0.0-chexpert.csv.gz" \
  "gs://mimic-cxr-jpg-2.1.0.physionet.org/mimic-cxr-2.0.0-metadata.csv.gz" \
  "gs://mimic-cxr-jpg-2.1.0.physionet.org/mimic-cxr-2.0.0-negbio.csv.gz" \
  "gs://mimic-cxr-jpg-2.1.0.physionet.org/mimic-cxr-2.0.0-split.csv.gz" \
  "gs://mimic-cxr-jpg-2.1.0.physionet.org/mimic-cxr-2.1.0-test-set-labeled.csv" \
  .
```

## MIMIC-IV

1. `cd <project_root>/datasets`
1. `wget -r -N -c -np --user username --password password --cut-dirs=1 -nH https://physionet.org/files/mimiciv/3.1/`

> Replace `username` and `password` above with your actual physionet.org credentials.

## MIMIC-IV-Note

1. `cd <project_root>/datasets`
1. `wget -r -N -c -np --user username --password password --cut-dirs=1 -nH https://physionet.org/files/mimic-iv-note/2.2/`

> Replace `username` and `password` above with your actual physionet.org credentials.

## MIMIC-IV-ECG

1. `cd <project_root>/datasets`
1. `wget -r -N -c -np --user username --password password --cut-dirs=1 -nH https://physionet.org/files/mimic-iv-ecg/1.0/`

> Replace `username` and `password` above with your actual physionet.org credentials.

## MIMIC-IV-ECG-Ext-ICD

1. `cd <project_root>/datasets`
1. `wget -r -N -c -np --user username --password password --cut-dirs=1 -nH https://physionet.org/files/mimic-iv-ecg-ext-icd-labels/1.0.1/`


## MIMIC-IV-ED

1. `cd <project_root>/datasets`
1. `wget -r -N -c -np --user username --password password --cut-dirs=1 -nH https://physionet.org/files/mimic-iv-ed/2.2/` 