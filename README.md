# bert-race-pytorch-lightening

bert for RACE with pytorch-lightening and transformer

adopted from [ASC2021-RACE](https://github.com/ASC-Competition/ASC2021-RACE)

## File Structure

```text
.
├── data
│   ├── RACE
│   │   ├── dev
│   │   ├── test
│   │   └── train
│   ├── RACEDataModule.py
│   └── RACELocalLoader.py
├── model
│   ├── bert-large-uncased
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── vocab.txt
│   ├── BertForRace.py
│   ├── BertLongAttention.py
│   ├── BertPooler.py
│   ├── CheckptEnsemble.py
│   ├── DCMNForRace.py
│   ├── FuseNet.py
│   └── SSingleMatchNet.py
├── plugins
│   ├── ApexDDP.py
│   └── ApexDDPAccelerator.py
├── result
│   └── asc01
├── hp_optimize.py
├── train.py
├── predict.py
├── README.md
├── requirements.txt
└── LICENSE
```

Please put the data and pre-trained model into `data` and `model` as above.

## Environment

```bash
pip install -r requirements.txt
```

You need to install `apex` separately

### On Cluster

```bash
scl enable devtoolset-9 bash
conda activte [env]
# then compile and install apex and other modules
```

> Install `horovod`
> 
> ```bash
> HOROVOD_NCCL_LIB=/usr/lib64/ HOROVOD_NCCL_INCLUDE=/usr/include/ HOROVOD_GPU_ALLREDUCE=NCCL HOROVOD_GPU_BROADCAST=NCCL HOROVOD_NCCL_LINK=SHARED pip install --no-cache-dir horovod
> ```

## TODO

- [x] PyTorch Lightening
- [x] Transformer
- [x] Refactor RACE Dataset Loader
  - [x] Use `datasets` from `transformer`
  - [x] Better Interface and Format
  - [x] Faster Data Loading (Using rust & multi-process)
  - [x] Cache Tokenized Results
  - [x] Custom Datasets (Local loading)
- [x] Mix Precision Training (Apex)
- [x] TensorBoard Logging
  - [x] Change Log Dir
- [x] Text Logging (Should be **same** as baseline code, ~~override pl original progress bar~~, will be done after ejection)
- [x] ~~Argparse~~ (Not that important)
- [x] Inference & Answer Saving
- [x] Hyper Parameter Tuning (Optuna)
  - [ ] More parameters
- [x] Parallelism
  - [x] FairShare
  - [x] ~~DeepSpeed~~ (Unstable)
- [x] Distributed (Will be done after ejection)
    - [x] DDP
    - [x] ~~Apex DDP~~ (Given up)
    - [x] ~~Apex + Horovod~~ (Given up)
- [x] ~~Cross Validation~~ (Useless)
- [x] ~~Data Augmentation~~ (Useless)
- [ ] Model Tweak
  - [x] DCMN (Buggy)
  - [x] Sentence Selection
  - [x] Sliding Window
  - [x] Rouge Score
  - [ ] Use features from previous layers
- [x] Model Ensemble (Buggy, will be done after ejection)
- [x] ~~Find Best Seed~~ (Useless, there will be new datasets and pre-trained model on-site)
- [x] Further Speedup Training Process
  - [x] ~~LongFormer~~ (Seems useless)
  - [x] ~~Nvidia Bert~~ (Will be done in ejection)
