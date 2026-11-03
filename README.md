# Diabetic Foot Assessment Using Thermal Asymmetry and Siamese Feature Learning

> **Research code repository** for a thermal-image-based framework that investigates bilateral foot asymmetry for diabetic foot assessment. The project is organised into three phases: masked thermal-image reconstruction, Siamese asymmetry learning, and spatial feature-difference visualisation.

**Publication status:** [The findings were presented in <CICBA 2026>; proceedings decision/publication status: <ACCEPTED>.]  
**Paper:** [WILL BE UPDATED]  
**Authors:** [Gulshan Katankar and Aadish Jain, P Saranya]

---

## Table of contents

- [Overview](#overview)
- [Research pipeline](#research-pipeline)
- [Phase 1: Thermal representation learning](#phase-1-thermal-representation-learning)
- [Dataset preparation](#dataset-preparation)
- [Phase 2: Bilateral asymmetry learning](#phase-2-bilateral-asymmetry-learning)
- [Phase 3: Spatial asymmetry visualisation](#phase-3-spatial-asymmetry-visualisation)
- [Experimental setup](#experimental-setup)
- [Results](#results)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Running the experiments](#running-the-experiments)
- [Reproducibility and limitations](#reproducibility-and-limitations)
- [Dataset and data availability](#dataset-and-data-availability)
- [Citation](#citation)
- [License](#license)
- [Disclaimer](#disclaimer)

---

## Overview

This project explores the use of thermal imaging and bilateral foot asymmetry for diabetic foot assessment. Rather than relying only on the absolute temperature of an individual foot, the Phase 2 model compares representations extracted from the left and right feet.

The implementation is divided into three phases:

| Phase | Objective | Main components |
|---|---|---|
| Phase 1 | Learn thermal-image representations through masked-image reconstruction | U-Net, ResNet-18 encoder, random patch masking |
| Phase 2 | Learn bilateral thermal asymmetry for binary classification | Siamese network, STN, pretrained encoder |
| Phase 3 | Visualise spatial feature differences between feet | Encoder feature maps, absolute feature difference, heatmap overlay |

The Phase 1 encoder is saved and subsequently loaded into the Siamese model in Phase 2. Phase 3 uses the trained Siamese model's encoder and alignment module to generate a spatial feature-difference map.

## Research pipeline

```text
Thermal CSV maps
      |
      v
PHASE 1: MASKED THERMAL RECONSTRUCTION
      |
      |  Random patch masking
      |  U-Net with ResNet-18 encoder
      |  Reconstruction using MSE loss
      |
      v
Pretrained thermal encoder weights
      |
      v
DATASET PREPARATION
      |
      |  Polygon annotations
      |  Thermal-image masking
      |  Individual foot extraction
      |  256 x 256 resizing
      |
      v
PHASE 2: BILATERAL ASYMMETRY LEARNING
      |
      |  Left/right foot pairs
      |  Horizontal flip and STN alignment
      |  Shared ResNet-18 encoder
      |  Absolute feature difference
      |  Binary classification
      |
      v
PHASE 3: SPATIAL ASYMMETRY VISUALISATION
      |
      |  Spatial encoder feature maps
      |  Absolute feature difference
      |  Upsampling and heatmap overlay
      |
      v
Visualisation of spatial feature asymmetry
```

---

## Phase 1: Thermal representation learning

### Objective

Phase 1 trains an encoder to represent the thermal structure of foot images through a masked-image reconstruction task.

The input thermal map is divided into patches. A proportion of these patches is randomly masked, and the model is trained to reconstruct the original thermal image.

### Dataset processing

The `INAOEThermalDataset` class:

1. Loads thermal maps stored as CSV files.
2. Excludes files whose names contain `LCA`, `LPA`, `MCA`, or `MPA`.
3. Converts the thermal matrix to `float32`.
4. Clips values to the range 20–40.
5. Normalises the values to the range [0, 1].
6. Resizes the map to 256 × 256 using cubic interpolation.
7. Adds a single channel dimension for thermal input.

### Masked reconstruction

Random contiguous patches are set to zero. The model receives the masked thermal image and attempts to reconstruct the original image.

| Configuration | Value |
|---|---:|
| Input resolution | 256 × 256 |
| Input channels | 1 |
| Architecture | U-Net |
| Encoder | ResNet-18 |
| Encoder initialisation | Random (no ImageNet weights) |
| Batch size | 16 |
| Epochs | 80 |
| Learning rate | 0.0001 |
| Validation split | 20% |
| Patch size | 16 × 16 |
| Mask ratio | 60% |
| Optimiser | Adam |
| Reconstruction loss | Mean Squared Error (MSE) |

The encoder weights from the model with the lowest validation loss are saved as `best_inaoe_encoder_v2.pth`.

### Phase 1: Reconstruction output

<!-- Replace the path below with the filename of your final exported reconstruction. -->
<p align="center">
  <img src="results/phase1/val_epoch_80.png" alt="Phase 1: original, masked, and reconstructed thermal foot image" width="850">
</p>

<p align="center"><em>Figure 1. Phase 1 thermal-image reconstruction. Replace this placeholder with the final reconstructed output.</em></p>

---

## Dataset preparation

The preprocessing pipeline converts polygon annotations into masked thermal foot images and extracts the two feet independently.

### Healthy samples

- Healthy annotations are read from the `healthy/RGB` directory.
- The corresponding thermal image is located in the `healthy/thermal` directory using the matching filename.
- Polygon coordinates from the JSON annotation are used to construct a binary mask.

### Diabetic samples

- Diabetic annotations and thermal images are read from the `diabetic/R0/thermal`, `diabetic/R1/thermal`, and `diabetic/R2/thermal` directories.
- Each JSON annotation is matched to an image with the same filename.

### Masking and foot extraction

1. A binary mask is created by filling the annotated polygons.
2. The mask is applied to the thermal image to suppress the background.
3. The two largest external contours are selected.
4. A bounding box is calculated for each contour.
5. A 10-pixel padding is added where image boundaries permit.
6. Each crop is resized to 256 × 256.
7. The crops are saved using `_L.png` and `_R.png` suffixes. These suffixes indicate image-side positions, not verified anatomical laterality.

### Expected processed directory

```text
STANDUP_Ready_Phase2/
├── healthy/
│   ├── <sample>_L.png
│   └── <sample>_R.png
├── diabetic_R0/
│   ├── <sample>_L.png
│   └── <sample>_R.png
├── diabetic_R1/
│   ├── <sample>_L.png
│   └── <sample>_R.png
└── diabetic_R2/
    ├── <sample>_L.png
    └── <sample>_R.png
```

The preprocessing script expects the original database to follow the folder conventions described above. Update `ROOT_DIR` and `OUTPUT_DIR` before running it.

**Important alignment requirement:** Healthy polygon annotations originate from RGB JSON files and are applied directly to thermal images in the supplied implementation. This assumes that RGB and thermal coordinates are spatially aligned. Verify registration and image dimensions before relying on these masks.

---

## Phase 2: Bilateral asymmetry learning

### Objective

Phase 2 investigates whether bilateral thermal feature differences can help distinguish healthy samples from diabetic samples.

Each sample consists of a left-foot image and a right-foot image. The two images are passed through a shared encoder, and their feature representations are compared.

### Labels and experimental modes

| Sample group | Label | Included when |
|---|---:|---|
| Healthy | 0 | Both R0 and R2 experiments |
| Diabetic R0 | 1 | `EXPERIMENT_MODE = "R0"` |
| Diabetic R2 | 1 | `EXPERIMENT_MODE = "R2"` |
| Diabetic R1 | — | Not included by the supplied Phase 2 loader |

R0 and R2 are treated as separate experimental configurations. Their clinical interpretation should follow the dataset's official documentation.

### Siamese architecture

The model contains three main components:

**1. Spatial Transformer Network (STN)**

The right-foot image is horizontally flipped and passed through an STN. The STN predicts an affine transformation and uses it to spatially transform the image.

**2. Shared ResNet-18 encoder**

The left and aligned right images are passed through the same ResNet-18 encoder initialised with the Phase 1 pretrained weights.

The final feature maps are globally averaged to obtain feature vectors:

\[
f_L = \operatorname{GAP}(E(x_L))
\]

\[
f_R = \operatorname{GAP}(E(T(x_R)))
\]

Here, \(E\) denotes the encoder and \(T\) denotes the flip-and-STN transformation.

**3. Feature-difference classifier**

The absolute difference between the two feature vectors is calculated:

\[
d = |f_L-f_R|
\]

The difference vector is passed through a classifier:

```text
Linear(512, 128)
ReLU
Linear(128, 1)
Sigmoid
```

The output is a sigmoid score trained against the binary sample label using binary cross-entropy loss.

### Data augmentation

Training images use the following Albumentations transformations:

- Random rotation: up to 10 degrees.
- Shift and scale: up to 5%.
- Random brightness and contrast adjustment: up to 0.1.
- Resize to 256 × 256.
- Normalisation with mean 0.5 and standard deviation 0.5.

Validation images are resized and normalised without the training augmentations.

### Training configuration

| Configuration | Value |
|---|---:|
| Input resolution | 256 × 256 |
| Encoder | ResNet-18 |
| Encoder initialisation | Phase 1 pretrained weights |
| Batch size | 8 |
| Epochs | 30 |
| Learning rate | 0.0001 |
| Optimiser | Adam |
| Loss | Binary Cross-Entropy |
| Training/validation split | 80% / 20% |
| Classification threshold | 0.5 |
| Model selection metric | Validation F1 score |

The best model is saved using the experiment-specific filename, for example `siamese_r2_best_metrics.pth`.

---

## Phase 3: Spatial asymmetry visualisation

### Objective

Phase 3 extracts spatial feature differences before global average pooling reduces the feature maps to a single feature vector.

The objective is to visualise where the left and right foot representations differ in feature space.

### Processing steps

1. Load the trained Siamese model.
2. Load both left and right thermal foot images.
3. Resize and normalise both images.
4. Horizontally flip the right-foot image.
5. Align it using the trained STN.
6. Extract the final encoder feature maps from both feet.
7. Calculate the absolute difference between the feature maps.
8. Average the difference across feature channels.
9. Upsample the resulting map to 256 × 256.
10. Normalise the heatmap and overlay it on the corresponding RGB foot image.

The feature-difference map is:

\[
D(i,j)=\frac{1}{C}\sum_{c=1}^{C}
|F_L(c,i,j)-F_R(c,i,j)|
\]

where \(C\) is the number of feature channels.

The resulting map represents **spatial feature-space asymmetry**, not a direct pixel-temperature difference or a clinically validated ulcer segmentation.

### Phase 3: Final visualisation

Add your final exported Phase 3 output image to `assets/phase3_asymmetry_heatmap.png`.

<!-- Replace the path below with the filename of your final exported Phase 3 image. -->
<p align="center">
  <img src="assets/phase3_asymmetry_heatmap.png" alt="Phase 3: RGB foot image, spatial asymmetry heatmap, and fused overlay" width="950">
</p>

<p align="center"><em>Figure 2. Phase 3 spatial feature-difference heatmap and fused visualisation. Replace this placeholder with the final output.</em></p>

---

## Experimental setup

The repository includes a separate ablation notebook evaluating four model configurations:

1. Baseline: no STN and random encoder initialisation.
2. STN only: STN enabled and random encoder initialisation.
3. Pretrained only: Phase 1 pretrained encoder without STN.
4. Proposed: STN enabled with Phase 1 pretrained encoder.

The ablation notebook uses a fixed random-split generator seed of 42. Its reported metrics are selected using the best validation F1 score during training.

### Ablation study results

**Numerical results are intentionally left blank until the experiments have been run and verified.** Populate these cells using the actual output of `ablation_study.ipynb`.

| Model configuration | STN | Phase 1 pretraining | Accuracy | F1-score | AUROC |
|---|:---:|:---:|---:|---:|---:|
| Baseline | No | No | TBD | TBD | TBD |
| STN only | Yes | No | TBD | TBD | TBD |
| Pretrained only | No | Yes | TBD | TBD | TBD |
| Proposed model | Yes | Yes | TBD | TBD | TBD |

### Classification metrics

The Phase 2 training/evaluation code calculates the following metrics at a 0.5 classification threshold:

| Metric | Description | R0 experiment | R2 experiment |
|---|---|---:|---:|
| Accuracy | Overall proportion of correct predictions | TBD | TBD |
| Precision | Proportion of positive predictions that are correct | TBD | TBD |
| Recall / Sensitivity | Proportion of actual positive samples detected | TBD | TBD |
| F1-score | Harmonic mean of precision and recall | TBD | TBD |
| AUROC | Area under the receiver operating characteristic curve | TBD | TBD |
| Specificity | Proportion of actual negative samples correctly identified | TBD | TBD |
| TN | True negatives | TBD | TBD |
| FP | False positives | TBD | TBD |
| FN | False negatives | TBD | TBD |
| TP | True positives | TBD | TBD |

**Metric reporting note:** The supplied Phase 2 code calculates accuracy, precision, recall, F1-score, AUROC, and a confusion matrix. Specificity and the individual confusion-matrix counts can be reported by deriving them from the confusion matrix. The supplied ablation notebook reports only accuracy, F1-score, and AUROC.

Do not fill the table with values from different splits or experiments. Report the evaluation protocol and sample counts alongside the metrics.

---

## Repository structure

A suggested repository layout is:

```text
diabetic-foot-thermal-asymmetry/
│
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── model.ipynb
│   └── ablation_study.ipynb
│
├── src/
│   ├── phase1_pretraining.py
│   ├── data_preparation.py
│   ├── phase2_siamese_training.py
│   └── phase3_heatmap_visualisation.py
│
├── assets/
│   ├── phase1_reconstruction.png
│   └── phase3_asymmetry_heatmap.png
│
├── results/
│   ├── ablation_results.csv
│   └── classification_metrics.csv
│
└── checkpoints/
    └── README.md
```

The `.ipynb` files can remain the primary executable research record. The `src/` scripts are optional refactored versions; only include them once they have been checked against the notebooks.

Large model checkpoints and datasets should not be committed without checking their licences, redistribution permissions, and repository size. The `checkpoints/README.md` can explain how authorised users can obtain the weights.

---

## Installation

The implementation uses Python and PyTorch, together with OpenCV, Albumentations, segmentation-models-pytorch, scikit-learn, and supporting libraries.

Create and activate a virtual environment:

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install opencv-python numpy pandas matplotlib tqdm
pip install albumentations scikit-learn
```

For GPU training, install a PyTorch build compatible with the CUDA version and GPU environment. Consult the official [PyTorch installation selector](https://pytorch.org/get-started/locally/).

A `requirements.txt` should be generated from the tested environment so that the exact package versions used for the experiments are recorded.

---

## Running the experiments

### 1. Phase 1: encoder pretraining

Place the authorised INAOE thermal CSV dataset in the expected directory and update the dataset path in the Phase 1 notebook.

```text
INAOE_Dataset/
└── ...
```

Run the Phase 1 training notebook. The best encoder weights are saved as:

```text
best_inaoe_encoder_v2.pth
```

### 2. Dataset preparation

Place the authorised STANDUP database in the expected folder structure. Update:

```python
ROOT_DIR = "./STANDUP_Database"
OUTPUT_DIR = "./STANDUP_Ready_Phase2"
```

Run the preprocessing script to generate the left/right foot crops.

### 3. Phase 2: Siamese training

Ensure that the Phase 1 encoder checkpoint is available at the configured path. Set the experiment mode:

```python
EXPERIMENT_MODE = "R2"  # Change to "R0" for the R0 experiment
```

Run the Phase 2 notebook. The best checkpoint is saved with an experiment-specific filename.

### 4. Ablation study

Open `ablation_study.ipynb`, verify the dataset and pretrained-weight paths, and run the four configurations. Record the actual printed results in the ablation table.

### 5. Phase 3: heatmap generation

Set the target sample, side, and model checkpoint in the Phase 3 code:

```python
TARGET_ID = "YOUR_SAMPLE_ID"
TARGET_SIDE = "R"  # "L" or "R"

MODEL_PATH = "siamese_r2_best_metrics.pth"
```

Ensure the matching thermal pair and corresponding RGB image are available. Run the visualisation code to generate the final output image.

---

## Reproducibility and limitations

The following points should be considered when reproducing or extending the experiments:

- **RGB-to-thermal registration:** Healthy RGB polygon coordinates are applied to thermal images. Registration must be verified.
- **Contour assumption:** Preprocessing selects the two largest external contours, assuming these correspond to the two feet.
- **Image-side naming:** `_L` and `_R` indicate image position in the supplied preprocessing code, not independently verified anatomical laterality.
- **Participant-level separation:** The supplied Phase 2 code uses an image-pair-level random 80/20 split. A participant-level split is preferable to prevent samples from the same participant appearing in both training and validation sets.
- **Augmentation:** The left and right images are transformed through separate calls to the augmentation pipeline. This can produce different random transformations for the two feet.
- **Ablation validation transform:** The ablation notebook changes `val_dataset.dataset.mode` after creating the random splits. Since both subsets reference the same underlying dataset, this also changes the transform used by the training subset. Verify or correct this before interpreting the ablation results.
- **Ablation reproducibility:** A fixed split seed is used, but model initialisation, data-loader shuffling, and other sources of randomness are not all explicitly seeded.
- **Phase 3 interpretation:** The heatmap shows feature-space differences. It should not be interpreted as ground-truth ulcer localisation or a clinically validated explanation without additional validation.
- **Clinical scope:** The experiments are research investigations and do not establish clinical diagnostic performance or readiness for clinical use.

---

## Dataset and data availability

The datasets used in this research are not included in this repository.

Users should obtain the datasets from their respective official sources and comply with their licences, access conditions, privacy requirements, and redistribution restrictions.

Do not upload identifiable patient information, restricted images, or dataset files without explicit permission.

---

## Citation

If you use this implementation in your research, please cite the associated research paper.

**Paper citation**

```text
[Add the final, publisher-approved bibliographic citation here.]
```

**DOI:** [Add DOI after publication, if assigned.]

A `CITATION.cff` file is provided so GitHub can display citation information for this repository. Update it with the final paper and software metadata when available.

---

## License

This repository uses the Apache License 2.0, subject to the rights of the authors and any applicable third-party materials.

See [LICENSE](LICENSE) for details.

---

## Disclaimer

This repository is intended for academic and research purposes. The implementation is an experimental research framework for analysing thermal-image patterns and bilateral foot asymmetry. It is not a medical device and must not be used as a substitute for professional medical assessment, diagnosis, or treatment.
