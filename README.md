# dpss-exp3-VC-BNF
Voice Conversion Experiments for THUHCSI Course : &lt;Digital Processing of Speech Signals>


## Set up environment

1. Install sox from http://sox.sourceforge.net/ or apt install sox

2. Install ffmpeg from https://www.ffmpeg.org/download.html#build-linux or apt install FFmpeg

3. Set up python environment through:
```bash
python3 -m venv /path/to/new/virtual/environment
source /path/to/new/virtual/environment/bin/activate
pip3 install -r dpss-exp3-VC-BNF/requirement_torch19.txt
# or you can use this if you prefer torch1.8 version
pip3 install -r dpss-exp3-VC-BNF/requirement_torch18.txt
```
Tips: You can also setup your own environment depends on cuda you have.
We recommend that you use pytorch 1.9.0 with the corresponding cuda version to avoid bug.
## Data Preparation
1. Download bzn/mst-male/mst-female corpus from here https://cloud.tsinghua.edu.cn/d/a30bef6b8d504e46bdb9/files/?p=%2Fsub_dataset.tar
2. Extract the dataset, and organize your data directories as follows:
```bash
dataset/
├── mst-female
├── mst-male
├── bzn
```
3. Download pretrained ASR model from here https://cloud.tsinghua.edu.cn/d/a30bef6b8d504e46bdb9/files/?p=%2Ffinal.pt
4. Move final.pt to ./pretrained_model/asr_model



## Any-to-One Voice Conversion Model

### Feature Extraction

```bash

CUDA_VISIBLE_DEVICES=0 python preprocess.py --data_dir /path/to/dataset/bzn --save_dir /path/to/save_data/bzn/

```

Your extracted features will be organized as follows:
```bash
bzn/
├── dev_meta.csv
├── f0s
│   ├── bzn_000001.npy
│   ├── ...
├── linears
│   ├── bzn_000001.npy
│   ├── ...
├── mels
│   ├── bzn_000001.npy
│   ├── ...
├── BNFs
│   ├── bzn_000001.npy
│   ├── ...
├── test_meta.csv
└── train_meta.csv
```

Tips: If you get 'Could not find a version for torch==1.9.0+cu111', run the following script to solve the problem. More details please refer to: https://jishuin.proginn.com/p/763bfbd5e54b.


```bash

pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

```

### Train

If you have GPU (one typical GPU is enough, nearly 1s/batch):
```bash
CUDA_VISIBLE_DEVICES=0 python train_to_one.py --model_dir ./exps/model_dir_to_bzn --test_dir ./exps/test_dir_to_bzn --data_dir /path/to/save_data/bzn/
```

If you have no GPU (nearly 5s/batch):

```bash
python train_to_one.py --model_dir ./exps/model_dir_to_bzn --test_dir ./exps/test_dir_to_bzn --data_dir /path/to/save_data/bzn/
```
### Inference

```bash
CUDA_VISIBLE_DEVICES=0 python inference_to_one.py --src_wav /path/to/source/xx.wav --ckpt ../exps/model_dir_to_bzn/bnf-vc-to-one-49.pt --save_dir ./test_dir/
```


## Any-to-Many Voice Conversion Model

### Feature Extraction

```bash
# In any-to-many VC task, we use all the above 3 speakers as the target speaker set.
CUDA_VISIBLE_DEVICES=0 python preprocess.py --data_dir /path/to/dataset/ --save_dir /path/to/save_data/exp3-data-all
```

Your extracted features will be organized as follows:
```bash
exp3-data-all/
├── dev_meta.csv
├── f0s
│   ├── bzn_000001.npy
│   ├── ...
├── linears
│   ├── bzn_000001.npy
│   ├── ...
├── mels
│   ├── bzn_000001.npy
│   ├── ...
├── BNFs
│   ├── bzn_000001.npy
│   ├── ...
├── test_meta.csv
└── train_meta.csv
```

### Train

If you have GPU (one typical GPU is enough, nearly 1s/batch):
```bash
CUDA_VISIBLE_DEVICES=0 python train_to_many.py --model_dir ./exps/model_dir_to_many --test_dir ./exps/test_dir_to_many --data_dir /path/to/save_data/exp3-data-all
```

If you have no GPU (nearly 5s/batch):

```bash
python train_to_many.py --model_dir ./exps/model_dir_to_many --test_dir ./exps/test_dir_to_many --data_dir /path/to/save_data/exp3-data-all
```
### Inference

```bash
# Here for inference, we use 'mst-male' as the target speaker. you can change the tgt_spk argument to any of the above 3 speakers. 
CUDA_VISIBLE_DEVICES=0 python inference_to_many.py --src_wav /path/to/source/*.wav --tgt_spk bzn/mst-female/mst-male --ckpt ./model_dir/bnf-vc-to-many-49.pt --save_dir ./test_dir/
```

## Assignment requirements
This project is a vanilla voice conversion system based on BNFs. 

When you encounter problems while finishing your project, search the issues first to see if there are similar problems. If there are no similar problems, you can create new issues and state you problems clearly.
