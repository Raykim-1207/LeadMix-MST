# LeadMix-MST: Robust Myocardial Infarction Detection from Minimal-Lead ECG Signals
This repository is the official implementation of "LeadMix-MST: Robust Myocardial Infarction Detection from Minimal-Lead ECG Signals, Submitted to IEEE Journal of Biomedical and Health Informatics (JBHI)".


# Overview

The 12-lead electrocardiogram (ECG) is the clinical gold standard for diagnosing myocardial infarction (MI), but its use in pre-hospital and resource-limited settings is hindered by hardware complexity and the expertise required for electrode placement. This study addresses a biomedical and health informatics challenge by enabling reliable MI screening from minimal-lead ECG signals suitable for mobile, telemedicine, and wearable applications.

<img width="1457" height="373" alt="Image" src="https://github.com/user-attachments/assets/f616ca61-d6ab-428f-b6b1-fc43798d4925" />

We propose LeadMix-MST, a lead-mixing multi-scale spatio-temporal deep learning architecture optimized for minimal-lead ECG analysis. LeadMix-MST structurally learns lead-aware contextual representations that capture spatio-temporal correlations between leads, comprising (i) MS-DSC for hierarchical temporal self-organizing camp expansion, (ii) LTA for latent space attention of inter-lead relationships, and (iii) LCM for adaptive feature scaling and alignment.

<img width="1500" height="428" alt="Image" src="https://github.com/user-attachments/assets/11068878-d390-417c-984e-1ab1b4ea10c0" />

# Requirements

To install requirements:

```python
pip install -r requirements.txt
```

# Dataset

1. Physikalisch-Technische Bundesanstalt (PTB)
 <br/> https://www.physionet.org/content/ptbdb/1.0.0/
2. PTB-XL
 <br/>https://physionet.org/content/ptb-xl/1.0.3/
3. Shandong Provincial Hospital (SPH)
 <br/>https://springernature.figshare.com/collections/A_large-scale_multi-label_12-lead_electrocardiogram_database_with_standardized_diagnostic_statements/5779802/1

To process dataset as mentioned above, run this command: For PTB:
```python
python preprocessing_PTB.py --path <path_to_data> --outpath <path_to_processed_data>
```

For PTB-XL:
```python
python preprocessing_PTBXL.py --path <path_to_data> --outpath <path_to_processed_data>
```

For SPH:
```python
python preprocessing_SPH.py --path <path_to_data> --outpath <path_to_processed_data>
```

- path_to_data: original dataset path
- path_to_processed_data: save path of processed dataset

# Training
To train the model in the paper, run this command:
```python
python train.py --path <path_to_data> --model_save_dir <drectory_saved_model> --outpath <path_to_processed_data> --model_name <model_name_saved>
```

# Evaluation
To evaluate the model in the paper, run this command:
```python
python evaluate.py --path <path_to_data> --model_save_dir <drectory_saved_model> --outpath <path_to_processed_data> --model_name <model_name_saved>
```

# Results
Using three public 12-lead ECG datasets, we conducted patient-level cross-dataset evaluations to assess generalization across different acquisition devices and populations. The proposed framework achieves an optimal balance between diagnostic accuracy, computational efficiency, and practical deployability. It shows significant translational potential for integration into clinical information systems and point-of-care decision support within medical informatics.

<img width="833" height="210" alt="Image" src="https://github.com/user-attachments/assets/3a2fa450-1f73-42b6-9bf4-2c8a7da10bec" />

<img width="994" height="318" alt="Image" src="https://github.com/user-attachments/assets/5829b217-8529-4284-883c-7d39d7703151" />

<img width="1000" height="604" alt="Image" src="https://github.com/user-attachments/assets/fb2998e9-9434-464a-b7b3-b072a10dcde6" />
