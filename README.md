# CAP Assignment 1 – Pneumonia Classification with ResNet-18

## Project Structure

```
CAP_Assgn_1/
├── train_evaluate_CNN.py   # Main training and evaluation script
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

## Requirements

```bash
pip install torch torchvision numpy matplotlib
```

## Dataset

This project uses the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.

### Download instructions

**Option 1 – Kaggle website (simplest)**

1. Go to https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
2. Click **Download** (you'll need a free Kaggle account)
3. Unzip the file — it will contain `train/`, `val/`, and `test/` folders
4. Move those three folders into the root of this project

**Option 2 – Kaggle API (command line)**

```bash
pip install kaggle

# Place your kaggle.json API key in ~/.kaggle/kaggle.json, then:
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip
```

After downloading, your directory should look like this:

```
CAP_Assgn_1/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```



## How to Run

### Task 1.1 – Train ResNet-18 from scratch (random initialisation)

```bash
python train_evaluate_CNN.py --mode 1
```

### Task 1.2 – Fine-tune pretrained ResNet-18 (ImageNet weights)

```bash
python train_evaluate_CNN.py --mode 2
```

### Optional arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | 1 | 1 = from scratch, 2 = fine-tune pretrained |
| `--num_epochs` | 30 | Number of training epochs |
| `--batch_size` | 32 | Batch size |
| `--learning_rate` | 0.001 | Initial learning rate |

Example with custom arguments:

```bash
python train_evaluate_CNN.py --mode 2 --num_epochs 20 --batch_size 32 --learning_rate 0.001
```

## Running on Google Colab (This is what I did to run this) 

1. Set runtime to GPU: `Runtime → Change runtime type → H100 GPU (If you are using Colab Pro)`
2. Upload `train/`, `val/`, `test/`, and `train_evaluate_CNN.py` to Google Drive
3. In a Colab notebook:

```python
from google.colab import drive
drive.mount('/content/drive')

%cd /content/drive/MyDrive/CAP_Assgn_1

!python train_evaluate_CNN.py --mode 1
!python train_evaluate_CNN.py --mode 2
```

## Outputs

After each run the following files are saved in the project directory:

| File | Description |
|---|---|
| `resnet18_mode1_curves.png` | Loss and accuracy plots for Task 1.1 |
| `resnet18_mode2_curves.png` | Loss and accuracy plots for Task 1.2 |
| `misclassified_mode1/` | Misclassified test images for Task 1.1 |
| `misclassified_mode2/` | Misclassified test images for Task 1.2 |

Misclassified images are organised into subfolders by error type:
- `true_NORMAL_pred_PNEUMONIA/` — false positives
- `true_PNEUMONIA_pred_NORMAL/` — false negatives

Per-class accuracy (NORMAL and PNEUMONIA separately) is printed to the console at the end of each run.
