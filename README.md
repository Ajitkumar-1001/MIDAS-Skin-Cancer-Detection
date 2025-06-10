# MIDAS: Multi-Modal Intelligent Dermoscopy Analysis System for Enhanced Skin Cancer Detection

<p align="center">
  <strong>Enhanced Skin Cancer Detection through Bidirectional Multi-Modal Deep Learning</strong>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="#datasets">Datasets</a> •
  <a href="#usage">Usage</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#results">Results</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## Project Information

**Course:** CSE 676: Deep Learning, Summer 2025  
**Institution:** University at Buffalo, SUNY  
**Authors:** Ajit Senthil Kumar, Manoj Maheshwar Jagadeesan  
**Instructor:** Professor Alina Vereshchaka  

## Overview

MIDAS is a novel bidirectional attention-based deep learning framework that combines dermoscopy images with clinical metadata for enhanced skin cancer classification. Unlike existing single-modal approaches, MIDAS enables mutual enhancement between visual and clinical modalities through sophisticated cross-attention mechanisms.

### Key Features

- **Multi-Modal Fusion**: Combines dermoscopy images with clinical metadata (age, sex, location, etc.)
- **Bidirectional Attention**: Novel cross-attention enabling mutual enhancement between modalities
- **Mobile-Ready**: Efficient architecture (37MB) suitable for smartphone deployment
- **Clinical Relevance**: Targets >90% sensitivity for cancer detection
- **Comprehensive Evaluation**: 5-fold cross-validation with clinical metrics

### Problem Statement

Current automated skin cancer detection systems rely solely on visual analysis of dermoscopy images, achieving 75-84% diagnostic accuracy. MIDAS addresses this limitation by integrating clinical metadata the way dermatologists do in practice, targeting 92-95% accuracy while maintaining high sensitivity crucial for cancer detection.

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- 8GB+ RAM
- 50GB+ storage for datasets

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/[username]/MIDAS-Skin-Cancer-Detection.git
cd MIDAS-Skin-Cancer-Detection
```

2. **Create virtual environment**
```bash
python -m venv midas_env
source midas_env/bin/activate  # On Windows: midas_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Install git-who for contribution tracking**
```bash
npm install -g git-who
git-who --setup
```

### Dependencies

```python
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
opencv-python>=4.5.0
albumentations>=1.3.0
wandb>=0.15.0
jupyter>=1.0.0
```

## Datasets

### HAM10000 Dataset

**Description:** 10,015 dermoscopic images across 7 skin lesion types  
**Classes:** MEL, BCC, AKIEC, BKL, DF, NV, VASC  
**Metadata:** Patient age, sex, lesion location

**Download Links:**
- [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
- [ISIC Archive](https://isic-archive.com/)
- [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

### PAD-UFES-20 Dataset

**Description:** 2,298 smartphone-collected clinical images  
**Features:** 21 clinical characteristics including Fitzpatrick skin type, lesion diameter  
**Download:** [Mendeley Data](https://data.mendeley.com/datasets/zr7vgbcyr2/1)

### Setup Instructions

1. **Create data directory**
```bash
mkdir data
cd data
```

2. **Download HAM10000**
```bash
# Download from Kaggle (requires kaggle API)
kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
unzip skin-cancer-mnist-ham10000.zip -d HAM10000/
```

3. **Download PAD-UFES-20**
```bash
# Manual download required from Mendeley link above
# Extract to data/PAD-UFES-20/
```

4. **Verify data structure**
```
data/
├── HAM10000/
│   ├── HAM10000_images_part_1/
│   ├── HAM10000_images_part_2/
│   ├── HAM10000_metadata.csv
│   └── HAM10000_segmentations_lesion_tschandl/
└── PAD-UFES-20/
    ├── images/
    └── metadata.csv
```

## Project Structure

```
MIDAS-Skin-Cancer-Detection/
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── LICENSE
│
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── midas.py              # Main MIDAS architecture
│   │   ├── encoders.py           # Image and metadata encoders
│   │   ├── attention.py          # Attention mechanisms
│   │   └── baselines.py          # Baseline models
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datasets.py           # Dataset classes
│   │   ├── preprocessing.py      # Data preprocessing
│   │   └── augmentation.py       # Data augmentation
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # Training loop
│   │   ├── losses.py             # Loss functions
│   │   └── metrics.py            # Evaluation metrics
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py             # Configuration management
│       ├── visualization.py      # Plotting and visualization
│       └── helpers.py            # Utility functions
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_training.ipynb
│   ├── 03_midas_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_attention_visualization.ipynb
│
├── configs/
│   ├── baseline_config.yaml
│   ├── midas_config.yaml
│   └── evaluation_config.yaml
│
├── scripts/
│   ├── train_baseline.py
│   ├── train_midas.py
│   ├── evaluate.py
│   └── preprocess_data.py
│
├── results/
│   ├── models/                   # Saved model checkpoints
│   ├── logs/                     # Training logs
│   ├── plots/                    # Generated plots
│   └── reports/                  # Evaluation reports
│
├── docs/
│   ├── architecture.md           # Detailed architecture docs
│   ├── training.md               # Training procedures
│   └── evaluation.md             # Evaluation methodology
│
└── assets/
    ├── midas_logo.png
    ├── architecture_diagram.png
    └── sample_results.png
```

## Usage

### Quick Start

1. **Preprocess datasets**
```bash
python scripts/preprocess_data.py --dataset HAM10000 --output data/processed/
```

2. **Train baseline model**
```bash
python scripts/train_baseline.py --config configs/baseline_config.yaml
```

3. **Train MIDAS model**
```bash
python scripts/train_midas.py --config configs/midas_config.yaml
```

4. **Evaluate results**
```bash
python scripts/evaluate.py --model results/models/midas_best.pth --dataset HAM10000
```

### Jupyter Notebooks

Explore the project interactively:

```bash
jupyter lab notebooks/
```

**Recommended Order:**
1. `01_data_exploration.ipynb` - Dataset analysis and visualization
2. `02_baseline_training.ipynb` - Train image-only baselines
3. `03_midas_training.ipynb` - Train MIDAS multi-modal model
4. `04_evaluation.ipynb` - Comprehensive evaluation and comparison
5. `05_attention_visualization.ipynb` - Attention mechanism analysis

### Configuration

Edit `configs/midas_config.yaml` to customize training:

```yaml
model:
  image_encoder: "efficientnet-b0"
  metadata_dim: 42
  fusion_dim: 768
  num_classes: 7
  
training:
  batch_size: 16
  learning_rate: 1e-4
  epochs: 100
  optimizer: "adamw"
  
data:
  image_size: 224
  augmentation: true
  cross_validation_folds: 5
```

## Architecture

MIDAS employs a sophisticated bidirectional attention mechanism for multi-modal fusion:

### Components

1. **Image Encoder**: EfficientNet-B0 backbone (1280-dim features)
2. **Metadata Encoder**: 3-layer MLP (256-dim features)
3. **Intra-Modal Self-Attention**: Focus on relevant features within each modality
4. **Bidirectional Cross-Attention**: Mutual enhancement between modalities
5. **Classification Head**: Multi-layer classifier with dropout

### Key Innovation

**Bidirectional Cross-Attention** enables:
- **Path 1**: Metadata guides visual attention to highlight diagnostically relevant image regions
- **Path 2**: Images guide metadata attention to emphasize clinically important patient characteristics

### Model Statistics

- **Total Parameters**: 9.14M
- **Model Size**: 37MB
- **Training Memory**: ~4.2GB (batch size 16)
- **Inference Time**: <100ms per image

## Results

### Expected Performance (Target)

| Model | Accuracy | AUC-ROC | Sensitivity | Specificity |
|-------|----------|---------|-------------|-------------|
| EfficientNet-B0 (baseline) | 87.2% | 0.924 | 85.1% | 89.3% |
| MetaNet (multi-modal) | 91.5% | 0.941 | 88.7% | 93.2% |
| ALBEF (current SOTA) | 94.1% | 0.943 | 92.3% | 95.1% |
| **MIDAS (proposed)** | **92-95%** | **0.945-0.950** | **>90%** | **>93%** |

### Evaluation Metrics

- **Clinical Metrics**: Sensitivity, Specificity (per-class and macro-averaged)
- **Standard Metrics**: Accuracy, Balanced Accuracy, F1-Score, AUC-ROC
- **Statistical Analysis**: 95% confidence intervals, significance testing

### Ablation Studies

- Bidirectional vs. unidirectional cross-attention
- Impact of different attention head configurations  
- Individual metadata feature importance analysis
- Comparison of fusion strategies (early vs. late fusion)

## Contributing

We welcome contributions to improve MIDAS. Please follow these guidelines:

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Commit: `git commit -m 'Add amazing feature'`
5. Push: `git push origin feature/amazing-feature`
6. Submit a pull request

### Code Standards

- Follow PEP 8 style guidelines
- Add docstrings to all functions and classes
- Include unit tests for new functionality
- Update documentation as needed

### Issue Reporting

Please use GitHub Issues to report bugs or request features. Include:
- Clear description of the issue
- Steps to reproduce (for bugs)
- Expected vs. actual behavior
- System information (OS, Python version, GPU)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Dataset Providers**: HAM10000 creators, PAD-UFES-20 team
- **ISIC Archive**: International Skin Imaging Collaboration
- **University at Buffalo**: Department of Computer Science and Engineering
- **Course Instructor**: Professor Alina Vereshchaka
- **Open Source Community**: PyTorch, scikit-learn, and other libraries

## Citation

If you use MIDAS in your research, please cite:

```bibtex
@misc{kumar2025midas,
  title={MIDAS: Multi-Modal Intelligent Dermoscopy Analysis System for Enhanced Skin Cancer Detection},
  author={Kumar, Ajit Senthil and Jagadeesan, Manoj Maheshwar},
  year={2025},
  school={University at Buffalo, SUNY},
  note={CSE 676: Deep Learning Final Project}
}
```

## Contact

- **Ajit Senthil Kumar**: senthil5@buffalo.edu
- **Manoj Maheshwar Jagadeesan**: manojmah@buffalo.edu

## Useful Links

- [Project Proposal](docs/proposal.pdf)
- [Architecture Documentation](docs/architecture.md)
- [Training Logs](https://wandb.ai/midas-team/skin-cancer-detection)
- [ISIC Archive](https://isic-archive.com/)
- [Course Website](https://engineering.buffalo.edu/computer-science-engineering/academics/graduate/graduate-course-descriptions.html)

---

<p align="center">
  University at Buffalo, SUNY | Department of Computer Science and Engineering
</p>
