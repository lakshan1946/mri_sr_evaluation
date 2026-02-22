# MRI Volume Extraction - Quick Start Guide

## What Does This Project Do?

This project **automatically analyzes brain MRI scans** and extracts volumetric measurements for 32 different brain structures. It's designed to evaluate super-resolution algorithms by comparing volumes between original and enhanced scans.

---

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Your MRI Scans
Place NIfTI files (.nii or .nii.gz) in the `inputs/` folder:
```bash
mkdir inputs
# Copy your .nii or .nii.gz files here
```

### 3. Run the Script
```bash
python extract_volumes.py
```

**Output**: `outputs/brain_volumes.csv` with volumetric measurements for all scans!

---

## 📊 What You'll Get

A CSV file with measurements like this:

| Filename | Left Cerebral White Matter | Left Cerebral Cortex | Left Hippocampus | ... |
|----------|---------------------------|---------------------|-----------------|-----|
| scan1.nii.gz | 318,158 mm³ | 378,654 mm³ | 5,458 mm³ | ... |
| scan2.nii.gz | 314,113 mm³ | 367,949 mm³ | 5,358 mm³ | ... |

**32 brain structures** measured automatically!

---

## 🧠 What Structures Are Measured?

### Cerebral Structures (L/R)
- White Matter
- Cortex
- Lateral Ventricles
- Thalamus
- Caudate, Putamen, Pallidum
- Hippocampus, Amygdala
- Accumbens Area

### Cerebellar Structures (L/R)
- White Matter
- Cortex

### Central Structures
- 3rd Ventricle
- 4th Ventricle
- Brain Stem
- Background

---

## ⚙️ Configuration Options

Edit these variables in `extract_volumes.py`:

```python
INPUT_DIR = './inputs'              # Where your MRI scans are
OUTPUT_DIR = './outputs'            # Where results go
SAVE_SEGMENTATION = False           # Set True to keep .nii segmentation files
```

---

## 🎯 Use Cases

### 1. Super-Resolution Evaluation
Compare volumes between:
- Original high-resolution scans
- Low-resolution scans
- Super-resolved outputs

### 2. Algorithm Comparison
Process outputs from different SR algorithms and compare accuracy

### 3. Clinical Research
Extract brain volumes for research studies with automated, consistent measurements

---

## ⏱️ Processing Time

- **GPU**: ~15 seconds per scan
- **CPU**: ~60 seconds per scan
- **Batch of 10 scans**: 2-10 minutes

---

## 🔧 Troubleshooting

### No NIfTI files found
```
Error: No NIfTI files found in ./inputs
```
**Solution**: Make sure your files end with `.nii` or `.nii.gz`

### Out of Memory
**Solution**: Edit `extract_volumes.py` and set:
```python
cropping=[192, 192, 192]  # Add to predict() call
```

### TensorFlow not found
**Solution**: Install dependencies:
```bash
pip install tensorflow==2.2.0 keras==2.3.1
```

---

## 📦 What's Included?

```
mri_sr_evaluation/
├── extract_volumes.py      ← Main script (run this!)
├── requirements.txt        ← Python packages to install
├── inputs/                 ← Put your MRI scans here
├── outputs/                ← Results appear here
│   └── brain_volumes.csv
└── synseg_model/           ← Pre-trained AI model
    └── synthseg_model/
        └── models/
            └── synthseg_1.0.h5
```

---

## 🤖 How It Works (Simple Explanation)

1. **Load MRI**: Reads your NIfTI brain scan
2. **AI Segmentation**: Uses a pre-trained neural network (SynthSeg) to label 32 brain regions
3. **Volume Calculation**: Counts voxels for each region and converts to mm³
4. **Export Results**: Saves everything to a CSV spreadsheet

### The AI Model (SynthSeg)
- Trained on **synthetic** brain images (not real scans!)
- Works on **any MRI contrast** (T1, T2, FLAIR, etc.)
- Handles **any resolution** (even low-quality clinical scans)
- **No preprocessing needed** (no bias correction, skull stripping, etc.)

---

## 📈 Analyzing Results

### In Python (pandas)
```python
import pandas as pd
df = pd.read_csv('outputs/brain_volumes.csv')

# Compare two scans
original = df[df['Filename'].str.contains('original')]
super_resolved = df[df['Filename'].str.contains('sr_')]

# Calculate difference
diff = super_resolved.iloc[0, 1:] - original.iloc[0, 1:]
print(diff)
```

### In Excel
1. Open `brain_volumes.csv`
2. Create pivot tables or charts
3. Calculate differences between scans

### In R
```r
df <- read.csv('outputs/brain_volumes.csv')
summary(df)
```

---

## 🎓 Academic Context

**Use Case**: Final Year Project on MRI Super-Resolution

**Research Question**: 
*"How accurately do super-resolution algorithms preserve brain structure volumes?"*

**Method**:
1. Start with high-resolution MRI scans
2. Artificially degrade them (downsampling, gaps, etc.)
3. Apply super-resolution algorithms
4. Compare volumes: Original vs. SR output
5. Determine which algorithm best preserves anatomical accuracy

---

## 📚 Need More Details?

See `PROJECT_DOCUMENTATION.md` for:
- Complete technical architecture
- Detailed component descriptions
- Mathematical formulations
- Advanced configuration options
- API reference
- Scientific background

---

## 🆘 Getting Help

### Common Questions

**Q: Can I use DICOM files?**
A: No, convert to NIfTI first using tools like `dcm2niix`

**Q: What MRI sequences work?**
A: All! T1, T2, FLAIR, PD, etc. SynthSeg is contrast-agnostic.

**Q: Do I need a GPU?**
A: No, but it's 4x faster. CPU works fine.

**Q: Can I process just one scan?**
A: Yes, put one file in `inputs/` and run normally.

**Q: How accurate is this?**
A: Validated on thousands of scans. Comparable to FreeSurfer but much faster.

---

## 📞 Support Resources

- **Full Documentation**: `PROJECT_DOCUMENTATION.md`
- **SynthSeg Paper**: [Medical Image Analysis, 2023](https://www.sciencedirect.com/science/article/pii/S1361841523000506)
- **GitHub**: [https://github.com/BBillot/SynthSeg](https://github.com/BBillot/SynthSeg)
- **FreeSurfer Wiki**: [SynthSeg Documentation](https://surfer.nmr.mgh.harvard.edu/fswiki/SynthSeg)

---

## ✅ Pre-Flight Checklist

Before running, ensure:
- [ ] Python 3.6 or 3.8 installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] MRI scans in `inputs/` folder
- [ ] Model weights present: `synseg_model/synseg_model/models/synthseg_1.0.h5`
- [ ] At least 8GB RAM available
- [ ] (Optional) NVIDIA GPU with CUDA 10.1

---

## 🎉 You're Ready!

Run this command and watch your volumes get extracted automatically:

```bash
python extract_volumes.py
```

Good luck with your research! 🧠🔬

---

**Last Updated**: February 22, 2026
