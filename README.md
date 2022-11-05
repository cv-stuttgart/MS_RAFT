# MS_RAFT

In this repository we release (for now) the inference code for our work:

> **[Multi-Scale RAFT: Combining Hierarchical Concepts for Learning-Based Optical Flow Estimation](https://dx.doi.org/10.1109/ICIP46576.2022.9898048)**<br/>
> _ICIP 2022_ <br/>
> Azin Jahedi, Lukas Mehl, Marc Rivinius and Andrés Bruhn

If you find our work useful please [cite via BibTeX](CITATIONS.bib).


## 🆕 Follow-Up Work

We improved the accuracy further by extending the method and applying a modified training setup.
Our new approach is called `MS_RAFT_plus` and won the [Robust Vision Challenge 2022](http://www.robustvision.net/).

The code is available on [GitHub](https://github.com/cv-stuttgart/MS_RAFT_plus). 


## Requirements

The code has been tested with PyTorch 1.10.2+cu113.
Install the required dependencies via
```
pip install -r requirements.txt
```

Alternatively you can also manually install the following packages in your virtual environment:
- `torch`, `torchvision`, and `torchaudio` (e.g., with `--extra-index-url https://download.pytorch.org/whl/cu113` for CUDA 11.3)
- `matplotlib`
- `scipy`
- `tensorboard`
- `opencv-python`
- `tqdm`
- `parse`


## Pre-Trained Checkpoints

You can download our pre-trained model from the [releases page](https://github.com/cv-stuttgart/MS_RAFT/releases/tag/v1.0.0).


## Datasets

Datasets are expected to be located under `./data` in the following layout:
```
./data
  ├── kitti15                   # KITTI 2015
  │  └── dataset
  │     ├── testing/...
  │     └── training/...
  └── sintel                    # Sintel
     ├── test/...
     └── training/...
```


## Running MS_RAFT

You can evaluate a trained model via
```Shell
python evaluate.py --model sintel.pth --dataset sintel
```
This needs about 12 GB of GPU VRAM on MPI Sintel images.

If your GPU has smaller capacity, please compile the CUDA correlation module (once) via:
```Shell
cd alt_cuda_corr && python setup.py install && cd ..
```
and then run:
```Shell
python evaluate.py --model sintel.pth --dataset sintel --cuda_corr
```
Using `--cuda_corr`, estimating the flow on MPI Sintel images needs about 4 GB of GPU VRAM.


## Acknowledgement

Parts of this repository are adapted from [RAFT](https://github.com/princeton-vl/RAFT) ([license](licenses/RAFT/LICENSE)).
We thank the authors for their excellent work.
