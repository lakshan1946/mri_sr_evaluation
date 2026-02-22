# MRI Super-Resolution Evaluation Project - Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Core Components](#core-components)
4. [Technical Deep Dive](#technical-deep-dive)
5. [Workflow and Usage](#workflow-and-usage)
6. [Dependencies and Setup](#dependencies-and-setup)
7. [Output Analysis](#output-analysis)
8. [Future Enhancements](#future-enhancements)

---

## 1. Project Overview

### 1.1 Purpose
This project is designed for **evaluating MRI super-resolution techniques** by performing automated brain segmentation and volumetric analysis on MRI scans. It leverages **SynthSeg**, a state-of-the-art deep learning model for domain-agnostic brain MRI segmentation.

### 1.2 Key Capabilities
- **Automated Brain Segmentation**: Segments brain MRI scans into 32 distinct anatomical regions
- **Volume Quantification**: Calculates precise volumetric measurements (in mm³) for each brain structure
- **Multi-Modal Compatibility**: Works with MRI scans of any contrast (T1, T2, FLAIR, etc.) and resolution
- **Batch Processing**: Processes multiple NIfTI files automatically with progress tracking
- **Research Ready**: Outputs standardized CSV reports suitable for statistical analysis

### 1.3 Project Context
Based on the workspace name and structure, this appears to be part of a **Final Year Project (FYP)** focused on evaluating the effectiveness of super-resolution algorithms on MRI brain scans. The volumetric analysis enables quantitative comparison between:
- Original high-resolution scans
- Super-resolved images from lower resolution inputs
- Various super-resolution algorithm outputs

---

## 2. System Architecture

### 2.1 Directory Structure
```
mri_sr_evaluation/
│
├── extract_volumes.py          # Main script for volume extraction
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules (ignores /inputs)
│
├── outputs/                    # Generated results
│   └── brain_volumes.csv       # Volumetric measurements
│
├── synseg_model/               # SynthSeg model package
│   └── synseg_model/
│       ├── models/
│       │   └── synthseg_1.0.h5 # Pre-trained weights
│       ├── data/
│       │   ├── labels table.txt
│       │   └── labels_classes_priors/
│       ├── SynthSeg/           # Core segmentation module
│       ├── ext/                # External utilities
│       └── scripts/            # Additional scripts
│
└── __pycache__/                # Python bytecode cache
```

### 2.2 Data Flow Architecture
```
┌─────────────────┐
│  Input MRI      │
│  (NIfTI files)  │
│  ./inputs/*.nii │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  extract_volumes.py     │
│  ┌────────────────────┐ │
│  │ 1. Load NIfTI      │ │
│  │ 2. Call SynthSeg   │ │
│  │ 3. Generate Seg    │ │
│  │ 4. Calculate Vols  │ │
│  └────────────────────┘ │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  SynthSeg Model         │
│  ┌────────────────────┐ │
│  │ U-Net CNN          │ │
│  │ (synthseg_1.0.h5)  │ │
│  │ Predict 32 labels  │ │
│  └────────────────────┘ │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Post-Processing        │
│  ┌────────────────────┐ │
│  │ Voxel counting     │ │
│  │ Volume calculation │ │
│  │ Label mapping      │ │
│  └────────────────────┘ │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Outputs                │
│  ├── brain_volumes.csv  │
│  └── seg_*.nii.gz       │
│      (optional)         │
└─────────────────────────┘
```

---

## 3. Core Components

### 3.1 Main Script: `extract_volumes.py`

#### 3.1.1 Key Functions

**`process_brain_mri(input_path, output_folder)`**
- Primary processing function for each MRI scan
- Steps:
  1. Runs SynthSeg inference
  2. Loads segmentation results
  3. Calculates voxel volume from NIfTI header
  4. Counts voxels per label
  5. Computes volumes (voxels × voxel_volume)
  6. Optionally cleans up intermediate files

**`get_voxel_volume(nii_img)`**
- Extracts voxel dimensions from NIfTI header
- Calculates voxel volume: `volume = x_dim × y_dim × z_dim`
- Returns volume in mm³

**`main()`**
- Orchestrates the entire batch processing pipeline
- Scans input directory for NIfTI files (.nii, .nii.gz)
- Processes each file with progress bar
- Aggregates results into pandas DataFrame
- Exports to CSV

#### 3.1.2 Configuration Parameters

```python
INPUT_DIR = './inputs'              # Source MRI directory
OUTPUT_DIR = './outputs'            # Destination for results
CSV_FILENAME = 'brain_volumes.csv' # Output file name
SAVE_SEGMENTATION = False           # Keep/delete .nii segmentations
```

#### 3.1.3 Label Mapping (FreeSurfer ColorLUT)

The script uses a comprehensive label dictionary with 32 brain structures:

| Label ID | Structure Name | Location |
|----------|---------------|----------|
| 0 | Background | - |
| 2-18 | Left Hemisphere Structures | Cerebral WM/Cortex, Ventricles, Cerebellum, etc. |
| 41-60 | Right Hemisphere Structures | Mirror of left structures |
| 14-16 | Midline Structures | 3rd/4th Ventricles, Brain Stem |

**Bilateral Structures**: 14 paired regions (WM, Cortex, Cerebellum, Thalamus, Caudate, Putamen, Pallidum, Hippocampus, Amygdala, Ventricles, Accumbens, Ventral DC)

### 3.2 SynthSeg Model

#### 3.2.1 Model Overview
**SynthSeg** (Synthesis-based Segmentation) is a revolutionary deep learning model that:
- Segments brain MRI scans **without requiring retraining** for different contrasts
- Trained on *synthetic* data generated from label maps
- Robust to any resolution (up to 10mm slice spacing)
- Works on preprocessed and non-preprocessed scans

#### 3.2.2 Key Features
1. **Domain Agnostic**: Works on T1, T2, FLAIR, PD, etc. without modification
2. **Resolution Invariant**: Handles heterogeneous resolutions (0.7mm - 10mm)
3. **Domain Randomization**: Trained with extreme augmentation for generalization
4. **Real-time Processing**: ~15s on GPU, ~60s on CPU per scan
5. **Output Resolution**: Always generates 1mm isotropic segmentations

#### 3.2.3 Model Architecture
- **Base Architecture**: 3D U-Net
- **Levels**: 5 hierarchical levels (configurable)
- **Convolutions per Level**: 2 layers (default)
- **Feature Maps**: 24 initial features, doubling per level
- **Weights**: Stored in `synthseg_1.0.h5` (Keras HDF5 format)

#### 3.2.4 Training Methodology
**Generative Approach**:
```
Label Maps → Image Generator → Synthetic MRI → U-Net Training
```

**Domain Randomization Strategy**:
- Randomizes intensity distributions from uniform priors
- Applies random spatial augmentations
- Simulates various MRI contrasts and artifacts
- No real MRI data used in training!

**Key Publications**:
1. *SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining* (Medical Image Analysis, 2023)
2. *Robust machine learning segmentation for large-scale analysis of heterogeneous clinical brain MRI datasets* (PNAS, 2023)

### 3.3 External Libraries (ext/)

#### 3.3.1 lab2im Package
**Purpose**: Labels-to-images generation for synthetic MRI creation

**Key Modules**:
- `image_generator.py`: On-the-fly synthetic image generation
- `edit_volumes.py`: Spatial transformations and augmentation
- `edit_tensors.py`: Tensor manipulation utilities
- `layers.py`: Custom Keras layers for augmentation
- `utils.py`: I/O operations, array handling, file management

#### 3.3.2 neuron Package
**Purpose**: Neural network utilities for medical imaging

**Key Modules**:
- `layers.py`: Custom neural network layers
- `models.py`: Model building utilities
- `utils.py`: Helper functions for medical image processing

### 3.4 BrainGenerator Class

Located in `SynthSeg/brain_generator.py`, this class encapsulates the synthetic data generation pipeline.

**Core Functionality**:
```python
class BrainGenerator:
    - Wraps labels_to_image_model
    - Generates synthetic MRI-label pairs
    - Applies augmentation on-the-fly
    - Feeds training data to U-Net
```

**Key Parameters**:
- **Spatial Augmentations**: Flipping, scaling (±20%), rotation (±15°), shearing, translation, non-linear deformations
- **Intensity Augmentations**: Bias field simulation, noise injection, contrast randomization
- **Resolution Simulation**: Randomizes resolution (isotropic up to 4mm, anisotropic up to 8mm)

---

## 4. Technical Deep Dive

### 4.1 Segmentation Labels (32 Structures)

Complete list from `labels table.txt`:

```
0   - Background
2   - Left Cerebral White Matter
3   - Left Cerebral Cortex
4   - Left Lateral Ventricle
5   - Left Inferior Lateral Ventricle
7   - Left Cerebellum White Matter
8   - Left Cerebellum Cortex
10  - Left Thalamus
11  - Left Caudate
12  - Left Putamen
13  - Left Pallidum
14  - 3rd Ventricle
15  - 4th Ventricle
16  - Brain Stem
17  - Left Hippocampus
18  - Left Amygdala
24  - CSF (SynthSeg 2.0+)
26  - Left Accumbens Area
28  - Left Ventral DC
41  - Right Cerebral White Matter
42  - Right Cerebral Cortex
43  - Right Lateral Ventricle
44  - Right Inferior Lateral Ventricle
46  - Right Cerebellum White Matter
47  - Right Cerebellum Cortex
49  - Right Thalamus
50  - Right Caudate
51  - Right Putamen
52  - Right Pallidum
53  - Right Hippocampus
54  - Right Amygdala
58  - Right Accumbens Area
60  - Right Ventral DC
```

**Note**: Label 24 (CSF) is only available in SynthSeg 2.0+. The project currently uses version 1.0.

### 4.2 Volume Calculation Formula

For each structure:
```
Volume (mm³) = Count_voxels × (Δx × Δy × Δz)
```

Where:
- `Count_voxels`: Number of voxels with specific label
- `Δx, Δy, Δz`: Voxel dimensions from NIfTI header

**Implementation**:
```python
zooms = nii_img.header.get_zooms()  # (Δx, Δy, Δz)
voxel_vol_mm3 = np.prod(zooms)      # mm³ per voxel
volume = count * voxel_vol_mm3
```

### 4.3 NIfTI File Format Handling

**NIfTI (Neuroimaging Informatics Technology Initiative)**:
- Standard format for medical imaging
- Supports 3D and 4D data
- Includes spatial metadata (affine transform, voxel dimensions)

**Library**: `nibabel` (Neuroimaging in Python)
```python
nii_img = nib.load(path)        # Load NIfTI file
data = nii_img.get_fdata()      # Extract voxel data
header = nii_img.header         # Access metadata
affine = nii_img.affine         # Get spatial transform
```

### 4.4 SynthSeg predict() Function

**Location**: `SynthSeg/predict.py`

**Function Signature**:
```python
def predict(path_images,
            path_segmentations,
            path_model,
            labels_segmentation,
            cropping=None,
            target_res=1.0,
            fast=False,
            robust=False,
            v1=False,
            ...)
```

**Key Parameters**:
- `path_images`: Input MRI scan path
- `path_segmentations`: Output segmentation path
- `path_model`: Path to trained weights (synthseg_1.0.h5)
- `labels_segmentation`: Label values array
- `cropping`: Optional crop size for memory constraints
- `target_res`: Output resolution (default 1mm isotropic)

**Processing Steps**:
1. Load and preprocess input image
2. Optionally resample to target resolution
3. Run U-Net inference
4. Apply optional test-time augmentation (TTA)
5. Perform topological corrections
6. Save segmentation as NIfTI

### 4.5 Advanced Features Available (Not Currently Used)

**SynthSeg 2.0 Features**:
- `--parc`: Cortical parcellation (subdivide cortex)
- `--robust`: Enhanced robustness for clinical scans
- `--ct`: CT scan compatibility
- `--qc`: Automated quality control scores
- `--vol`: Direct volume calculation (bypasses manual computation)
- `--post`: Save probability maps instead of hard labels

---

## 5. Workflow and Usage

### 5.1 Preparation Phase

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Download Model Weights** (if not present):
   - Access [UCL Dropbox](https://liveuclac-my.sharepoint.com/:f:/g/personal/rmappmb_ucl_ac_uk/EtlNnulBSUtAvOP6S99KcAIBYzze7jTPsmFk2_iHqKDjEw?e=rBP0RO)
   - Download `synthseg_1.0.h5`
   - Place in `synseg_model/synseg_model/models/`

3. **Prepare Input Data**:
   - Create `./inputs/` directory
   - Copy MRI scans in NIfTI format (.nii or .nii.gz)
   - Supported: T1, T2, FLAIR, PD, etc.

### 5.2 Execution Phase

**Basic Usage**:
```bash
python extract_volumes.py
```

**Expected Output**:
```
Found 9 files. Starting SynthSeg...
100%|████████████| 9/9 [02:15<00:00, 15.06s/it]
Success! Results saved to: ./outputs/brain_volumes.csv
```

**Processing Time**:
- GPU: ~15 seconds per scan
- CPU: ~60 seconds per scan
- Batch of 10 scans: ~2-10 minutes

### 5.3 Configuration Options

**Keep Segmentation Files**:
```python
SAVE_SEGMENTATION = True  # Saves seg_*.nii.gz files
```

**Memory Constrained Systems**:
```python
cropping = [192, 192, 192]  # Reduce memory usage
```

**Custom Directories**:
```python
INPUT_DIR = '/path/to/custom/inputs'
OUTPUT_DIR = '/path/to/custom/outputs'
```

### 5.4 Error Handling

The script includes robust error handling:
```python
try:
    result = process_brain_mri(full_path, OUTPUT_DIR)
    all_results.append(result)
except Exception as e:
    traceback.print_exc()
    print(f"Error processing {f}: {e}")
    # Continues with next file
```

**Common Issues**:
- Missing input directory → Creates with message
- Corrupted NIfTI files → Skips with error log
- Out of memory → Use `cropping` parameter

---

## 6. Dependencies and Setup

### 6.1 Python Version
- **Required**: Python 3.6 or 3.8
- **Recommended**: Python 3.8 for better compatibility

### 6.2 Core Dependencies

From `requirements.txt`:
```
tensorflow>=2.0      # Deep learning framework
neurite              # Medical imaging utilities
nibabel              # NIfTI file I/O
numpy                # Numerical computing
pandas               # Data manipulation and CSV export
tqdm                 # Progress bars
```

### 6.3 GPU Support (Optional but Recommended)

**For GPU Acceleration**:
- CUDA 10.0 (Python 3.6) or CUDA 10.1 (Python 3.8)
- cuDNN 7.6.5
- tensorflow-gpu (automatically installed with conda)

**Installation with Conda (Recommended)**:
```bash
# Python 3.8 with GPU
conda create -n synthseg_38 python=3.8 \
    tensorflow-gpu=2.2.0 keras=2.3.1 \
    nibabel matplotlib \
    -c anaconda -c conda-forge

conda activate synthseg_38
```

**Installation with Pip**:
```bash
pip install tensorflow-gpu==2.2.0 keras==2.3.1 \
    protobuf==3.20.3 numpy==1.23.5 \
    nibabel==5.0.1 matplotlib==3.6.2 \
    pandas tqdm
```

### 6.4 System Requirements

**Minimum**:
- CPU: Quad-core processor
- RAM: 8 GB
- Storage: 2 GB for model and dependencies

**Recommended**:
- CPU: 8+ cores
- RAM: 16 GB
- GPU: NVIDIA GPU with 4+ GB VRAM
- Storage: 10+ GB for inputs/outputs

---

## 7. Output Analysis

### 7.1 CSV Structure

The output file `brain_volumes.csv` contains:
- **Row**: One MRI scan
- **Column 1**: Filename
- **Columns 2-33**: Volume measurements (mm³) for each structure

**Example**:
```csv
Filename,Background,Left Cerebral White Matter,...
scan1.nii.gz,6988631.0,318158.0,...
scan2.nii.gz,7021741.0,314113.0,...
```

### 7.2 Sample Data Analysis

From the existing output, we observe:

**Subject**: 100206 (Human Connectome Project ID)
**Scans**: Multiple variants of T2-weighted scans
- Original high-resolution
- Gap-simulated (thickness variations: 3mm, 4mm, 5mm)
- In-plane downsampled (factors: 1x, 2x)
- Thick-slice acquisitions (3mm, 5mm)

**Volumetric Comparison Example**:

| Scan Type | Left Cerebral WM (mm³) | Left Cerebral Cortex (mm³) | Left Hippocampus (mm³) |
|-----------|----------------------|--------------------------|----------------------|
| Original HR | 318,158 | 378,654 | 5,458 |
| Gap 3mm | 314,113 | 367,949 | 5,358 |
| Gap 4mm | 301,741 | 378,781 | 5,115 |
| Gap 5mm | 295,950 | 383,380 | 5,293 |
| Thick 5mm | 300,254 | 379,132 | 5,016 |

**Observations**:
- White matter volumes decrease with degraded quality
- Cortex volumes show more variability
- Smaller structures (hippocampus) more affected by resolution

### 7.3 Statistical Analysis Applications

**Possible Analyses**:

1. **Volumetric Accuracy**:
   - Compare super-resolved vs. original volumes
   - Calculate mean absolute error (MAE)
   - Compute Dice coefficients for structures

2. **Structure-Specific Performance**:
   - Which structures are better preserved?
   - Small structures (hippocampus, amygdala) vs. large (cerebral WM)

3. **Algorithm Comparison**:
   - Compare multiple super-resolution methods
   - Statistical significance testing (paired t-tests)

4. **Clinical Relevance**:
   - Assess if volumetric differences exceed clinical thresholds
   - Evaluate reliability for diagnosis/monitoring

**Suggested Tools**:
- Python: SciPy, statsmodels, seaborn for visualization
- R: Statistical modeling and ANOVA
- Excel: Quick comparisons and charts

---

## 8. Future Enhancements

### 8.1 Potential Improvements

**1. Direct Volume Export**:
```python
# Use SynthSeg's built-in volume calculation
predict(..., vol=output_csv_path)
```
Benefits: More efficient, one less file I/O operation

**2. Quality Control Integration**:
```python
predict(..., qc=qc_scores_path)
```
Benefits: Automatic detection of failed segmentations

**3. Cortical Parcellation**:
```python
predict(..., parc=True)
```
Benefits: 100+ cortical regions instead of 2

**4. Probabilistic Segmentations**:
```python
predict(..., post=posteriors_path)
```
Benefits: Uncertainty quantification, soft segmentations

**5. Robust Mode for Low-Quality Scans**:
```python
predict(..., robust=True)
```
Benefits: Better performance on clinical/degraded data

### 8.2 Advanced Analysis Pipeline

**Proposed Extension**:
```
1. Volume Extraction (current)
   ↓
2. Statistical Comparison Module
   - Compute errors vs. ground truth
   - Generate comparison plots
   ↓
3. Visualization Dashboard
   - Interactive plots (Plotly/Dash)
   - 3D rendering of segmentations
   ↓
4. Report Generation
   - Automated LaTeX/Markdown reports
   - Publication-ready figures
```

### 8.3 Integration Improvements

**Docker Containerization**:
```dockerfile
FROM tensorflow/tensorflow:2.2.0-gpu
COPY . /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python", "extract_volumes.py"]
```
Benefits: Reproducibility, easy deployment

**Web Interface**:
- Flask/Django web app
- Upload → Process → Download results
- Real-time progress tracking

**Batch Job System**:
- SLURM/PBS integration for HPC clusters
- Array jobs for parallel processing
- Automatic resource allocation

### 8.4 Scientific Extensions

**1. Multi-Modal Analysis**:
- Process T1, T2, FLAIR simultaneously
- Cross-modal consistency checks

**2. Longitudinal Tracking**:
- Registration of follow-up scans
- Volumetric change analysis over time

**3. Population Studies**:
- Aggregate statistics across cohorts
- Normative database comparison

**4. Super-Resolution Evaluation Metrics**:
- Automated calculation of PSNR, SSIM
- Structure-specific perceptual quality metrics

---

## Appendix A: Mathematical Formulation

### Volume Calculation

For a segmentation $S$ with label $l$:

$$V_l = \sum_{i,j,k} \mathbb{1}_{S_{ijk}=l} \cdot \Delta x \cdot \Delta y \cdot \Delta z$$

Where:
- $\mathbb{1}_{S_{ijk}=l}$ is the indicator function (1 if voxel has label $l$, else 0)
- $\Delta x, \Delta y, \Delta z$ are voxel dimensions in mm

### U-Net Architecture

The SynthSeg model uses a 5-level U-Net:

$$\text{U-Net}: \mathbb{R}^{H \times W \times D} \rightarrow \mathbb{R}^{H \times W \times D \times C}$$

Where:
- Input: 3D MRI volume
- Output: $C=32$ probability maps (softmax)
- Final segmentation: $\arg\max_c P(c|x)$

---

## Appendix B: Troubleshooting

### Issue 1: TensorFlow Version Mismatch
**Error**: `ModuleNotFoundError: No module named 'tensorflow'`
**Solution**: Install exact version: `pip install tensorflow==2.2.0`

### Issue 2: GPU Not Detected
**Error**: `Could not load dynamic library 'libcudart.so.10.1'`
**Solution**: 
- Verify CUDA installation: `nvidia-smi`
- Install cuDNN: [NVIDIA Developer](https://developer.nvidia.com/cudnn)
- Use CPU fallback: Set `CUDA_VISIBLE_DEVICES=""`

### Issue 3: Memory Overflow
**Error**: `ResourceExhaustedError: OOM when allocating tensor`
**Solution**:
```python
predict(..., cropping=[160, 160, 160])
```

### Issue 4: Incorrect Output Resolution
**Problem**: Segmentation doesn't overlay on input
**Solution**: Use `--resample` flag to save co-registered images

---

## Appendix C: References

### Key Papers
1. Billot et al. (2023). "SynthSeg: Segmentation of brain MRI scans of any contrast and resolution without retraining." *Medical Image Analysis*. [Link](https://www.sciencedirect.com/science/article/pii/S1361841523000506)

2. Billot et al. (2023). "Robust machine learning segmentation for large-scale analysis of heterogeneous clinical brain MRI datasets." *PNAS*. [Link](https://www.pnas.org/doi/full/10.1073/pnas.2216399120)

### Software Resources
- **SynthSeg GitHub**: https://github.com/BBillot/SynthSeg
- **FreeSurfer**: https://surfer.nmr.mgh.harvard.edu/
- **NiBabel Documentation**: https://nipy.org/nibabel/
- **TensorFlow**: https://www.tensorflow.org/

### Related Tools
- **FreeSurfer SynthSeg**: Available in FreeSurfer 7.2+
- **MATLAB SynthSeg**: Medical Imaging Toolbox (2022b+)
- **FSL (FMRIB Software Library)**: Alternative segmentation tools
- **ANTs**: Advanced registration and segmentation

---

## Document Metadata

**Author**: GitHub Copilot (AI Documentation Assistant)
**Date Created**: February 22, 2026
**Version**: 1.0
**Project**: MRI Super-Resolution Evaluation
**Institution**: FYP (Final Year Project)
**Last Updated**: February 22, 2026

---

## License

This project uses **SynthSeg** which is licensed under Apache 2.0.

```
Copyright 2020 Benjamin Billot

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```

---

**End of Documentation**
