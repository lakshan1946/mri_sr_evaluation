# Visual Architecture Diagrams

## System Overview Diagram

```
╔════════════════════════════════════════════════════════════════════════════╗
║                    MRI SUPER-RESOLUTION EVALUATION SYSTEM                   ║
╚════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────┐
│                              INPUT STAGE                                  │
│  ┏━━━━━━━━━━━━━━━┓  ┏━━━━━━━━━━━━━━━┓  ┏━━━━━━━━━━━━━━━┓              │
│  ┃ Original HR   ┃  ┃ Low-Res Scan  ┃  ┃ Super-Resolved┃              │
│  ┃ MRI Scan      ┃  ┃ MRI Scan      ┃  ┃ MRI Scan      ┃              │
│  ┃ (.nii.gz)     ┃  ┃ (.nii.gz)     ┃  ┃ (.nii.gz)     ┃              │
│  ┗━━━━━━━┯━━━━━━━┛  ┗━━━━━━━┯━━━━━━━┛  ┗━━━━━━━┯━━━━━━━┛              │
│          │                  │                  │                        │
│          └────────────┬─────┴──────────────────┘                        │
└───────────────────────┼──────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         PROCESSING PIPELINE                               │
│                                                                           │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Stage 1: NIfTI Loading & Preprocessing                         │   │
│   │  ┌────────────┐   ┌────────────┐   ┌─────────────┐            │   │
│   │  │ Read File  │ → │ Load Array │ → │ Extract     │            │   │
│   │  │ (nibabel)  │   │ (get_fdata)│   │ Metadata    │            │   │
│   │  └────────────┘   └────────────┘   └─────────────┘            │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                 │                                         │
│                                 ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Stage 2: SynthSeg Inference                                    │   │
│   │  ┌────────────┐   ┌────────────┐   ┌─────────────┐            │   │
│   │  │ Resample   │ → │ CNN        │ → │ Argmax      │            │   │
│   │  │ to 1mm     │   │ Prediction │   │ (32 labels) │            │   │
│   │  └────────────┘   └────────────┘   └─────────────┘            │   │
│   │                                                                  │   │
│   │  Model: U-Net (5 levels, 24 features)                          │   │
│   │  Weights: synthseg_1.0.h5 (pre-trained)                        │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                 │                                         │
│                                 ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  Stage 3: Volume Calculation                                    │   │
│   │  ┌────────────┐   ┌────────────┐   ┌─────────────┐            │   │
│   │  │ Count      │ → │ Multiply   │ → │ Create      │            │   │
│   │  │ Voxels     │   │ by Voxel   │   │ Dictionary  │            │   │
│   │  │ per Label  │   │ Volume     │   │ (32 values) │            │   │
│   │  └────────────┘   └────────────┘   └─────────────┘            │   │
│   │                                                                  │   │
│   │  Formula: Volume = count × (Δx × Δy × Δz)                      │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                 │                                         │
└─────────────────────────────────┼─────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                            OUTPUT STAGE                                   │
│  ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓  │
│  ┃              brain_volumes.csv (spreadsheet)                      ┃  │
│  ┃  ┌──────────────┬─────────┬──────────┬───────────┬────────────┐  ┃  │
│  ┃  │ Filename     │ Struct1 │ Struct2  │ Struct3   │ ...        │  ┃  │
│  ┃  ├──────────────┼─────────┼──────────┼───────────┼────────────┤  ┃  │
│  ┃  │ original.nii │ 318,158 │ 378,654  │ 5,458     │ ...        │  ┃  │
│  ┃  │ sr_out.nii   │ 315,200 │ 375,123  │ 5,321     │ ...        │  ┃  │
│  ┃  │ ...          │ ...     │ ...      │ ...       │ ...        │  ┃  │
│  ┃  └──────────────┴─────────┴──────────┴───────────┴────────────┘  ┃  │
│  ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛  │
│                                                                           │
│  Optional: seg_*.nii.gz (segmentation masks if SAVE_SEGMENTATION=True)  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## U-Net Architecture (SynthSeg Model)

```
INPUT: 3D MRI Volume (any size) → Resampled to 1mm isotropic
│
▼
┌─────────────────────────────────────────────────────────────────┐
│                        ENCODER (Contracting Path)                │
├─────────────────────────────────────────────────────────────────┤
│  Level 1: [N × N × N] × 24 features                            │
│    Conv3D(3×3×3) → BatchNorm → ReLU                            │
│    Conv3D(3×3×3) → BatchNorm → ReLU                            │
│    ↓ MaxPool(2×2×2)                                            │
├─────────────────────────────────────────────────────────────────┤
│  Level 2: [N/2 × N/2 × N/2] × 48 features                      │
│    Conv3D(3×3×3) → BatchNorm → ReLU                            │
│    Conv3D(3×3×3) → BatchNorm → ReLU                            │
│    ↓ MaxPool(2×2×2)                                            │
├─────────────────────────────────────────────────────────────────┤
│  Level 3: [N/4 × N/4 × N/4] × 96 features                      │
│    Conv3D(3×3×3) → BatchNorm → ReLU                            │
│    Conv3D(3×3×3) → BatchNorm → ReLU                            │
│    ↓ MaxPool(2×2×2)                                            │
├─────────────────────────────────────────────────────────────────┤
│  Level 4: [N/8 × N/8 × N/8] × 192 features                     │
│    Conv3D(3×3×3) → BatchNorm → ReLU                            │
│    Conv3D(3×3×3) → BatchNorm → ReLU                            │
│    ↓ MaxPool(2×2×2)                                            │
├─────────────────────────────────────────────────────────────────┤
│  Level 5 (Bottleneck): [N/16 × N/16 × N/16] × 384 features     │
│    Conv3D(3×3×3) → BatchNorm → ReLU                            │
│    Conv3D(3×3×3) → BatchNorm → ReLU                            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DECODER (Expanding Path)                  │
├─────────────────────────────────────────────────────────────────┤
│  Level 4': [N/8 × N/8 × N/8] × 192 features                    │
│    ↑ UpConv(2×2×2)                                             │
│    Concatenate with Encoder Level 4                            │
│    Conv3D(3×3×3) → BatchNorm → ReLU                            │
│    Conv3D(3×3×3) → BatchNorm → ReLU                            │
├─────────────────────────────────────────────────────────────────┤
│  Level 3': [N/4 × N/4 × N/4] × 96 features                     │
│    ↑ UpConv(2×2×2)                                             │
│    Concatenate with Encoder Level 3                            │
│    Conv3D(3×3×3) → BatchNorm → ReLU                            │
│    Conv3D(3×3×3) → BatchNorm → ReLU                            │
├─────────────────────────────────────────────────────────────────┤
│  Level 2': [N/2 × N/2 × N/2] × 48 features                     │
│    ↑ UpConv(2×2×2)                                             │
│    Concatenate with Encoder Level 2                            │
│    Conv3D(3×3×3) → BatchNorm → ReLU                            │
│    Conv3D(3×3×3) → BatchNorm → ReLU                            │
├─────────────────────────────────────────────────────────────────┤
│  Level 1': [N × N × N] × 24 features                           │
│    ↑ UpConv(2×2×2)                                             │
│    Concatenate with Encoder Level 1                            │
│    Conv3D(3×3×3) → BatchNorm → ReLU                            │
│    Conv3D(3×3×3) → BatchNorm → ReLU                            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                         OUTPUT LAYER                             │
│    Conv3D(1×1×1) → 32 channels (one per class)                 │
│    Softmax Activation                                            │
│    Argmax → Final Segmentation                                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
OUTPUT: 3D Segmentation Mask [N × N × N] with values 0-60 (32 classes)
```

---

## Training Strategy: Domain Randomization

```
╔══════════════════════════════════════════════════════════════════╗
║              SYNTHSEG TRAINING PIPELINE (SIMPLIFIED)             ║
╚══════════════════════════════════════════════════════════════════╝

Step 1: Load Label Map
┌─────────────────────┐
│  Label Map          │  Example: Sagittal slice
│  ┌───────────────┐  │  ┌─────────────────┐
│  │ 0 0 0 0 0 0   │  │  │ Background: 0   │
│  │ 0 2 3 3 3 0   │  │  │ WM (L): 2       │
│  │ 0 2 3 3 42 0  │  │  │ Cortex (L): 3   │
│  │ 0 17 16 53 0  │  │  │ Cortex (R): 42  │
│  │ 0 0 0 0 0 0   │  │  │ Hippocampus: 17 │
│  └───────────────┘  │  │ Brainstem: 16   │
└─────────────────────┘  └─────────────────┘
         │
         ▼
Step 2: Random Intensity Assignment
┌──────────────────────┐
│ For each structure:  │  Example intensities:
│ μ ~ Uniform(0,255)   │  • Background: 20
│ σ ~ Uniform(1,30)    │  • WM: 180
│                      │  • Cortex: 120
│ I = N(μ, σ²)         │  • Hippocampus: 110
└──────────────────────┘
         │
         ▼
Step 3: Spatial Augmentation
┌──────────────────────────────┐
│ • Rotation: ±15°             │
│ • Scaling: ±20%              │
│ • Shearing                   │
│ • Non-linear deformation     │
└──────────────────────────────┘
         │
         ▼
Step 4: Resolution Simulation
┌──────────────────────────────┐
│ • Random anisotropic blur    │
│ • Simulate thick slices      │
│ • Downsample                 │
│ • Resample to 1mm            │
└──────────────────────────────┘
         │
         ▼
Step 5: MRI Artifact Simulation
┌──────────────────────────────┐
│ • Bias field (smooth grad)   │
│ • Gaussian noise             │
│ • Gamma correction           │
└──────────────────────────────┘
         │
         ▼
┌──────────────────────┐
│  Synthetic MRI       │  Looks like real T1/T2/FLAIR!
│  ┌────────────────┐  │  ┌─────────────────┐
│  │ [Gray levels]  │  │  │ But generated   │
│  │ [vary by       │  │  │ from labels!    │
│  │ [structure]    │  │  └─────────────────┘
│  └────────────────┘  │
└──────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│   U-Net Training             │
│   Loss = CrossEntropy        │
│   Optimizer = Adam           │
│   Epochs = 100,000           │
└──────────────────────────────┘
         │
         ▼
    Trained Model
    (synthseg_1.0.h5)
```

---

## Data Flow Through extract_volumes.py

```
START
  │
  ├─→ Scan './inputs/' directory
  │
  ├─→ Filter files: *.nii, *.nii.gz
  │
  ├─→ If no files found → Print error → EXIT
  │
  ├─→ Initialize: all_results = []
  │
  └─→ FOR EACH FILE:
      │
      ├─→ FUNCTION: process_brain_mri()
      │   │
      │   ├─→ Build output path: "seg_" + filename
      │   │
      │   ├─→ CALL: SynthSeg.predict()
      │   │   │
      │   │   ├─→ Load NIfTI
      │   │   ├─→ Preprocess (normalize, resize)
      │   │   ├─→ CNN inference
      │   │   ├─→ Save segmentation.nii.gz
      │   │   └─→ Return
      │   │
      │   ├─→ Load segmentation: nibabel.load()
      │   │
      │   ├─→ Extract voxel data: get_fdata()
      │   │
      │   ├─→ Calculate voxel volume:
      │   │   zooms = header.get_zooms()
      │   │   voxel_vol = zooms[0] × zooms[1] × zooms[2]
      │   │
      │   ├─→ Count voxels per label:
      │   │   unique, counts = np.unique(data, return_counts=True)
      │   │
      │   ├─→ Calculate volumes:
      │   │   FOR EACH label in LABEL_MAP:
      │   │       volume[label] = count[label] × voxel_vol
      │   │
      │   ├─→ Optional: Delete segmentation file
      │   │
      │   └─→ RETURN: {Filename: ..., Structure1: vol1, ...}
      │
      ├─→ Append result to all_results[]
      │
      └─→ NEXT FILE

  │
  ├─→ Create DataFrame: pd.DataFrame(all_results)
  │
  ├─→ Reorder columns: ['Filename', Structure1, Structure2, ...]
  │
  ├─→ Export CSV: df.to_csv('brain_volumes.csv')
  │
  └─→ Print success message

END
```

---

## Volume Calculation Detail

```
╔══════════════════════════════════════════════════════════════════╗
║                    VOLUME CALCULATION PROCESS                    ║
╚══════════════════════════════════════════════════════════════════╝

Input: Segmentation Mask (3D array)
Example: 256 × 256 × 256 voxels

Step 1: Extract Metadata
┌────────────────────────────┐
│  NIfTI Header              │
│  ┌──────────────────────┐  │
│  │ pixdim[1] = 1.0 mm   │  │
│  │ pixdim[2] = 1.0 mm   │  │
│  │ pixdim[3] = 1.0 mm   │  │
│  └──────────────────────┘  │
└────────────────────────────┘
            │
            ▼
  Voxel Volume = 1.0 × 1.0 × 1.0 = 1.0 mm³

Step 2: Count Voxels per Label
┌────────────────────────────┐
│  Segmentation Array        │
│  [0, 0, 2, 2, 3, 3, ...]   │
│                            │
│  np.unique(array,          │
│    return_counts=True)     │
│                            │
│  Returns:                  │
│  labels:  [0, 2, 3, ...]   │
│  counts: [1M, 300k, 350k]  │
└────────────────────────────┘

Step 3: Calculate Volumes
┌────────────────────────────────────────┐
│  For each label:                       │
│                                        │
│  Label 0 (Background):                 │
│    1,000,000 voxels × 1.0 mm³/voxel   │
│    = 1,000,000 mm³                     │
│                                        │
│  Label 2 (Left Cerebral WM):          │
│    300,000 voxels × 1.0 mm³/voxel     │
│    = 300,000 mm³                       │
│                                        │
│  Label 3 (Left Cerebral Cortex):      │
│    350,000 voxels × 1.0 mm³/voxel     │
│    = 350,000 mm³                       │
│                                        │
│  ... (repeat for all 32 labels)       │
└────────────────────────────────────────┘

Step 4: Store in Dictionary
┌────────────────────────────────────────┐
│  {                                     │
│    'Filename': 'scan.nii.gz',         │
│    'Background': 1000000.0,           │
│    'Left Cerebral White Matter':      │
│        300000.0,                       │
│    'Left Cerebral Cortex': 350000.0,  │
│    ...                                 │
│  }                                     │
└────────────────────────────────────────┘

Output: Dictionary of volumes (mm³)
```

---

## Brain Structure Topology

```
╔══════════════════════════════════════════════════════════════════╗
║                 BRAIN STRUCTURES (32 REGIONS)                    ║
╚══════════════════════════════════════════════════════════════════╝

                    SUPERIOR VIEW (Axial Slice)
                    
    ┌────────────────────────────────────────────┐
    │                                            │
    │     ╔══════════════════════════╗           │
    │     ║  Left Hemisphere (L)    ║           │
    │     ║  ┌──────────────────┐   ║           │
    │     ║  │ Cerebral Cortex  │   ║  (Midline)│
    │     ║  │ (Gray Matter)    │   ║     │     │
    │     ║  │  [Label: 3]      │   ║     │     │
    │     ║  └──────────────────┘   ║     ▼     │
    │     ║  ┌──────────────────┐   ║           │
    │     ║  │ Cerebral WM      │   ║  ╔════════╗
    │     ║  │ (White Matter)   │   ║  ║ 3rd    ║
    │     ║  │  [Label: 2]      │   ║  ║ Vent.  ║
    │     ║  │                  │   ║  ║ [14]   ║
    │     ║  │  ┌───────────┐   │   ║  ╚════════╝
    │     ║  │  │ Thalamus  │   │   ║           │
    │     ║  │  │ [10]      │   │   ║           │
    │     ║  │  └───────────┘   │   ║           │
    │     ║  │  ┌───────────┐   │   ║           │
    │     ║  │  │ Lateral   │   │   ║   Right   │
    │     ║  │  │ Ventricle │   │   ║ Hemisphere│
    │     ║  │  │ [4]       │   │   ║   (Mirror)│
    │     ║  │  └───────────┘   │   ║           │
    │     ║  └──────────────────┘   ║           │
    │     ╚══════════════════════════╝           │
    │                                            │
    └────────────────────────────────────────────┘

                    CORONAL VIEW (Front)
                    
           Left              Midline             Right
            │                   │                  │
            ▼                   ▼                  ▼
    ┌──────────────┬───────────────────┬──────────────┐
    │  Cerebral    │                   │  Cerebral    │
    │  Cortex [3]  │                   │  Cortex [42] │
    ├──────────────┤                   ├──────────────┤
    │  Cerebral    │    Brain Stem     │  Cerebral    │
    │  WM [2]      │      [16]         │  WM [41]     │
    │              │                   │              │
    │  ┌────────┐  │   ┌─────────┐    │  ┌────────┐  │
    │  │Thalamus│  │   │4th Vent │    │  │Thalamus│  │
    │  │  [10]  │  │   │  [15]   │    │  │  [49]  │  │
    │  └────────┘  │   └─────────┘    │  └────────┘  │
    │              │                   │              │
    │  ┌────────┐  │                   │  ┌────────┐  │
    │  │Hippocp.│  │                   │  │Hippocp.│  │
    │  │  [17]  │  │                   │  │  [53]  │  │
    │  └────────┘  │                   │  └────────┘  │
    └──────────────┴───────────────────┴──────────────┘
                INFERIOR (Cerebellum below)

```

---

## File Format: brain_volumes.csv Structure

```
Row 1 (Header):
┌──────────┬────────────┬────────────┬────────────┬─────┬────────────┐
│ Filename │ Background │ Left Cbl WM│ Left Ctx   │ ... │ Right VDC  │
└──────────┴────────────┴────────────┴────────────┴─────┴────────────┘
    │           │            │            │                    │
    │           │            │            │                    │
    ▼           ▼            ▼            ▼                    ▼
  String      float64     float64      float64              float64
            (mm³)        (mm³)        (mm³)                (mm³)

Row 2+ (Data):
┌────────────────┬────────────┬────────────┬────────────┬─────┬────────┐
│ scan1.nii.gz   │ 6988631.0  │ 318158.0   │ 378654.0   │ ... │ 5043.0 │
├────────────────┼────────────┼────────────┼────────────┼─────┼────────┤
│ scan2.nii.gz   │ 7021741.0  │ 314113.0   │ 367949.0   │ ... │ 5583.0 │
├────────────────┼────────────┼────────────┼────────────┼─────┼────────┤
│ scan3.nii.gz   │ 7034255.0  │ 301741.0   │ 378781.0   │ ... │ 5075.0 │
└────────────────┴────────────┴────────────┴────────────┴─────┴────────┘

Total Columns: 33 (1 filename + 32 structures)
Total Rows: N + 1 (N scans + 1 header)

File Size: ~N KB for N scans (very compact!)
Format: UTF-8 encoded, comma-separated
Compatible: Excel, Python (pandas), R, SPSS, etc.
```

---

## Super-Resolution Evaluation Workflow

```
╔══════════════════════════════════════════════════════════════════╗
║         COMPLETE SR ALGORITHM EVALUATION WORKFLOW                ║
╚══════════════════════════════════════════════════════════════════╝

Phase 1: Data Preparation
┌─────────────────────────────────────┐
│  High-Resolution Ground Truth       │
│  • Acquire: 3T/7T scanner           │
│  • Resolution: 0.7-1.0 mm isotropic │
│  • Format: NIfTI (.nii.gz)          │
└────────────┬────────────────────────┘
             │
             ├─→ [Create Degraded Version]
             │   • Downsample to 3mm
             │   • Add noise
             │   • Simulate artifacts
             │
             ▼
      ┌────────────────┐     ┌────────────────┐
      │ HR Original    │     │ LR Degraded    │
      │ (Ground Truth) │     │ (Algorithm In) │
      └────────────────┘     └────────┬───────┘
             │                        │
             │                        ▼
             │              ┌──────────────────┐
             │              │ SR Algorithm     │
             │              │ • SRCNN          │
             │              │ • EDSR           │
             │              │ • Real-ESRGAN    │
             │              └────────┬─────────┘
             │                       │
             │                       ▼
             │              ┌──────────────────┐
             │              │ SR Output        │
             │              │ (Enhanced)       │
             │              └────────┬─────────┘
             │                       │
             └───────┬───────────────┘
                     │
Phase 2: Volume Extraction
                     ▼
        ┌────────────────────────────┐
        │  Run extract_volumes.py    │
        │  on 3 inputs:              │
        │  1. HR original            │
        │  2. LR degraded            │
        │  3. SR output              │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Obtain 3 CSV files:       │
        │  • volumes_hr.csv          │
        │  • volumes_lr.csv          │
        │  • volumes_sr.csv          │
        └────────────┬───────────────┘
                     │
Phase 3: Comparative Analysis
                     ▼
        ┌────────────────────────────┐
        │  Calculate Errors:         │
        │                            │
        │  Error_SR = |V_SR - V_HR|  │
        │             ───────────────│
        │                  V_HR      │
        │                            │
        │  Error_LR = |V_LR - V_HR|  │
        │             ───────────────│
        │                  V_HR      │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │  Statistical Tests:        │
        │  • Paired t-test           │
        │  • Wilcoxon signed-rank    │
        │  • Effect size (Cohen's d) │
        └────────────┬───────────────┘
                     │
Phase 4: Visualization
                     ▼
        ┌────────────────────────────┐
        │  Generate Plots:           │
        │  • Bar charts (errors)     │
        │  • Box plots (distribution)│
        │  • Scatter (SR vs HR)      │
        │  • Bland-Altman plots      │
        └────────────┬───────────────┘
                     │
Phase 5: Reporting
                     ▼
        ┌────────────────────────────┐
        │  Conclusions:              │
        │  ✓ SR reduces error by X%  │
        │  ✓ Best preserved: WM, Ctx │
        │  ✓ Worst preserved: Hipp   │
        │  ✓ Clinical viability: Y/N │
        └────────────────────────────┘
```

---

## Performance Optimization Strategies

```
╔══════════════════════════════════════════════════════════════════╗
║                    OPTIMIZATION OPTIONS                          ║
╚══════════════════════════════════════════════════════════════════╝

1. GPU ACCELERATION
   ┌──────────────────────────────────────────┐
   │  Current: Sequential CPU/GPU processing  │
   │  Time: ~60s/scan (CPU), ~15s (GPU)       │
   └──────────────────────────────────────────┘
            Optimization ▼
   ┌──────────────────────────────────────────┐
   │  • Ensure GPU visible                    │
   │  • Batch processing on GPU               │
   │  Time: ~10s/scan (GPU batched)           │
   └──────────────────────────────────────────┘

2. MULTIPROCESSING (CPU)
   ┌──────────────────────────────────────────┐
   │  Current: Serial for-loop                │
   │  Time: N × 60s for N scans               │
   └──────────────────────────────────────────┘
            Optimization ▼
   ┌──────────────────────────────────────────┐
   │  from multiprocessing import Pool        │
   │  pool.map(process_scan, file_list)       │
   │  Time: (N × 60s) / cores                 │
   └──────────────────────────────────────────┘

3. MEMORY MANAGEMENT
   ┌──────────────────────────────────────────┐
   │  Current: Load full volume               │
   │  Memory: ~4-8 GB per scan                │
   └──────────────────────────────────────────┘
            Optimization ▼
   ┌──────────────────────────────────────────┐
   │  • Use cropping parameter                │
   │  • Process sub-volumes                   │
   │  Memory: ~2-3 GB                         │
   └──────────────────────────────────────────┘

4. I/O OPTIMIZATION
   ┌──────────────────────────────────────────┐
   │  Current: Save/load intermediate .nii.gz │
   │  Time: ~2-3s per file                    │
   └──────────────────────────────────────────┘
            Optimization ▼
   ┌──────────────────────────────────────────┐
   │  • Use predict() with direct volume out  │
   │  • Skip intermediate save                │
   │  Time: ~0s (no disk I/O)                 │
   └──────────────────────────────────────────┘

COMBINED SPEEDUP:
┌──────────────────────────────────────────────┐
│  Baseline: 60s/scan × 100 scans = 100 min   │
│  Optimized: 10s/scan × 100 scans = 17 min   │
│  Speedup: ~6x faster                         │
└──────────────────────────────────────────────┘
```

---

## Error Analysis Framework

```
╔══════════════════════════════════════════════════════════════════╗
║              VOLUMETRIC ERROR ANALYSIS PYRAMID                   ║
╚══════════════════════════════════════════════════════════════════╝

Level 1: Raw Errors
┌────────────────────────────────────────────────┐
│  For each structure i:                         │
│                                                │
│  Absolute Error: |V_SR[i] - V_HR[i]| (mm³)    │
│  Relative Error: |V_SR[i] - V_HR[i]| / V_HR[i]│
│  Signed Error: V_SR[i] - V_HR[i] (over/under) │
└────────────────────────────────────────────────┘
                      │
                      ▼
Level 2: Aggregate Statistics
┌────────────────────────────────────────────────┐
│  Mean Absolute Error (MAE)                     │
│  Root Mean Squared Error (RMSE)                │
│  Mean Relative Error (MRE)                     │
│  Median Absolute Deviation (MAD)               │
└────────────────────────────────────────────────┘
                      │
                      ▼
Level 3: Structure Categories
┌────────────────────────────────────────────────┐
│  Large structures:   Error_large               │
│    (WM, Cortex)                                │
│                                                │
│  Small structures:   Error_small               │
│    (Hippocampus, Amygdala)                     │
│                                                │
│  Compare: Error_small > Error_large?           │
└────────────────────────────────────────────────┘
                      │
                      ▼
Level 4: Clinical Relevance
┌────────────────────────────────────────────────┐
│  Is |Error| < Clinical_Threshold?              │
│                                                │
│  Hippocampus: Threshold = 5%                   │
│  Ventricles:  Threshold = 10%                  │
│  Total Brain: Threshold = 3%                   │
│                                                │
│  ✓ Pass: SR clinically acceptable              │
│  ✗ Fail: SR not reliable                       │
└────────────────────────────────────────────────┘
```

---

**End of Visual Diagrams**
**Use these for presentations, documentation, and conceptual understanding!**
