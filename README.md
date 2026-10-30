# VGG6 on CIFAR-10

Explores a 6-weight-layer VGG (4 conv + 2 fully-connected) on CIFAR-10 while varying
activation function, optimizer, batch size, epochs and learning rate. All runs are
logged to Weights & Biases (W&B).

## Repository layout
| File | Purpose |
|---|---|
| `config.py` | default hyper-parameters |
| `data.py` | CIFAR-10 download, normalisation, augmentation, train/val/test split |
| `model.py` | the VGG6 network |
| `train.py` | training loop (+ W&B logging, checkpoint saving, final test accuracy) |
| `evaluate.py` | evaluation function; script to re-evaluate a saved model |
| `run_experiments.py` | all controlled experiments for Question 2 |
| `sweep.yaml` | optional random W&B sweep |
| `best_model.pt` | trained weights of the best configuration |

## Environment
- Python 3.10+ (fill in exact version: `python --version`)
- Packages: `pip install -r requirements.txt`
  Exact versions used for the reported results: _(paste output of_ `pip freeze | grep -i -E "torch|torchvision|wandb|numpy|matplotlib"` _here)_
- GPU strongly recommended (Any GPU or Google Colab T4). Works on CPU but is very slow.

## Seed configuration
Default seed is **42** (`--seed`). It is used for Python, NumPy, PyTorch (CPU+CUDA), the
data-loader shuffling and the train/validation split. `cudnn.deterministic=True`,
`cudnn.benchmark=False`. Same seed + same hardware/software gives the same numbers;
a different GPU model can change results in the last decimals.

## How to reproduce
```bash
    Clone the git repository using following command
a. git clone <this-repo-url> && cd <repo>
b. pip install -r requirements.txt
c. wandb login                                   # paste here your API key from wandb.ai/authorize

# 1) One single run
python train.py --activation relu --optimizer adam --lr 0.001 --batch_size 128 --epochs 20

# 2) All Experiments in one go
python run_experiments.py


# 3) Verify the saved model / make local curves
python evaluate.py --ckpt best_model.pt
```
If you dont have W&B account? Add `--no_wandb` to `train.py` / `run_experiments.py`.


