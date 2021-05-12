# bert-race-pytorch-lightning

BERT for RACE with pytorch-lightning and transformer

Implemented [DCMN](https://arxiv.org/abs/1908.11511) ([reference code](https://github.com/Qzsl123/dcmn)) and [DUMA](https://arxiv.org/abs/2001.09415)

This repo is for experimental purposes. In order to achieve the best performance on distributed systems, we ejected the code from pytorch-lightening to native pytorch and changed the model from the one implemented by huggingface to [Nvidia's](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT).

For the ejected version, check out [bert-race-nvidia](https://github.com/iamNCJ/bert-race-nvidia).

## File Structure

```text
.
├── data
│   ├── RACE
│   │   ├── dev
│   │   ├── test
│   │   └── train
│   ├── RACEDataModule.py
│   ├── RACEDataModuleForALBERT.py
│   └── RACELocalLoader.py
├── model
│   ├── bert-large-uncased
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── vocab.txt
│   ├── ALBERTForRace.py
│   ├── BertForRace.py
│   ├── BertLongAttention.py
│   ├── BertPooler.py
│   ├── CheckptEnsemble.py
│   ├── DCMNForRace.py
│   ├── DUMAForRace.py
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

## Features

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
  - [x] More parameters (Will be done in ejection)
- [x] Parallelism
  - [x] FairShare
  - [x] ~~DeepSpeed~~ (Unstable)
- [x] Distributed (Will be done after ejection)
    - [x] DDP
    - [x] ~~Apex DDP~~ (Given up)
    - [x] ~~Apex + Horovod~~ (Given up)
- [x] ~~Cross Validation~~ (Useless)
- [x] ~~Data Augmentation~~ (Useless)
- [x] Model Tweak
  - [x] DCMN (***Bad test result (acc around 60 only, far lower than the paper's result)*** && buggy now, I'm not going to debug it anymore, if anyone wants to use it, please checkout a working commit [#1df19a5](https://github.com/iamNCJ/bert-race-pytorch-lightening/tree/1df19a519e5113a4985cb8a10e586754941d0a33))
  - [x] DUMA
  - [x] Sentence Selection (Bad result)
  - [x] Sliding Window (Bad result)
  - [x] Rouge Score (small improvement on short sequences)
  - [x] ~~Use features from previous layers~~ (Useless)
- [x] Model Ensemble (Buggy, will be done after ejection)
- [x] ~~Find Best Seed~~ (Useless, there will be new datasets and pre-trained model on-site)
- [x] Further Speedup Training Process
  - [x] ~~LongFormer~~ (Seems useless)
  - [x] ~~Nvidia Bert~~ (Will be done in ejection)
