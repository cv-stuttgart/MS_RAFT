# MS_RAFT_plus

In this repository we release (for now) the inference code for our work:

> **[High Resolution Multi-Scale RAFT (Robus Vision Challenge 2022)](https://arxiv.org/abs/2210.16900)**<br/>
> _Robust Vision Challenge 2022_ <br/>
> Azin Jahedi, Maximilian Luz, Lukas Mehl, Marc Rivinius and Andrés Bruhn

If you find our work useful please [cite via BibTeX](CITATIONS.bib).

This work builds upon [`MS_RAFT`](https://github.com/cv-stuttgart/MS_RAFT).


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

You can download our pre-trained model from the [releases page](https://github.com/cv-stuttgart/MS_RAFT_plus/releases/tag/v1.0.0).


## Datasets

Datasets are expected to be located under `./data` in the following layout:
```
./data
  ├── kitti15                   # KITTI 2015
  │  └── dataset
  │     ├── testing/...
  │     └── training/...
  ├── middlebury                # Middlebury
  │  ├── test/...
  │  │  └── img/...
  │  └── training/...
  │     ├── flow/...
  │     └── img/...
  ├── sintel                    # Sintel
  │  ├── test/...
  │  └── training/...
  └── viper                     # Viper
     ├── test/img/...
     └── val
        ├── flow/...
        └── img/...
```


## Running MS_RAFT_plus

For running `MS_RAFT_plus` on MPI Sintel images you need about 4 GB of GPU VRAM.
 
To compile the CUDA correlation module run the following once:
```Shell
cd alt_cuda_corr && python setup.py install && cd ..
```
And then you can evaluate the pre-trained model via:
```Shell
python evaluate.py --model mixed.pth --dataset sintel --cuda_corr
```
Note that the above-mentioned (with `--cuda_corr`) code performs on-demand cost computation and does not pre-compute the cost volume, because such computation is very memory intensive on high resolutions.


## Acknowledgement

Parts of this repository are adapted from [RAFT](https://github.com/princeton-vl/RAFT) ([license](licenses/RAFT/LICENSE)).
We thank the authors for their excellent work.
