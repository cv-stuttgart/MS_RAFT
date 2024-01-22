# MS_RAFT

In this repository we release the code for our work:

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
  ├── fc                        # FlyingChairs
  │  └── data/...
  ├── fth                       # FlyingThings3D
  │  ├── frames_cleanpass/...
  │  ├── frames_finalpass/...
  │  └── optical_flow/...
  ├── HD1k                      # HD1K
  │  ├── hd1k_flow_gt/...
  │  └── hd1k_input/...
  ├── kitti15                   # KITTI 2015
  │  └── dataset
  │     ├── testing/...
  │     └── training/...
  ├── sintel                    # Sintel
  │  ├── test/...
  │  └── training/...
  └── viper                     # VIPER
     ├── train
     │  ├── flow/...
        └── img/...
     └── val
        ├── flow/...
        └── img/...
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

## Training MS_RAFT

You can train MS_RAFT via:
```Shell
python train.py --config config/train.json
```
This will create a folder called checkpoints if it does not exist. It will save the logs and checkpoints under a folder with the same name as the name attribute in the config file `config.json`, in this case: MS_RAFT_test.

Notes regarding the code:
- An important feature that is added here is a more space-saving checkpointing. The checkpoints are saved permanently only at the end of each training stage: After training is completed on Chairs, Things and Sintel. There are some intermediate checkpoints that are saved every VAL_FREQ iterations, but they get over-written by the new ones not to save unnecessary files.

- If you want to change some training parameters within a specific training stage, in checkpoint.txt in the experiment directory, you can set the attribute "phase" to the phase (stage) you want to train (1 for training on Things and 2 for Sintel), "current_steps" to 0 if you want to train that stage from the begining, "newer" to the path of the checkpoint in checkpoint.txt inside your experiment forlder. You can set "older" to null. Note that there is no need to change checkpoint.txt if you just want to train from scratch. In that case, you just start training with the proper config file like shown above.

- Importantly, rarely the loss becomes NaN and if the network continues training it might diverge, therefore the program checks for that and exits if the loss becomes NaN. In such cases just start training again with the same config file and without changing checkpoint.txt. The porgram loads the last checkpoint automatically and resumes training from the last saved checkpoint.

## Acknowledgement

Parts of this repository are adapted from [RAFT](https://github.com/princeton-vl/RAFT) ([license](licenses/RAFT/LICENSE)).
We thank the authors for their excellent work.
