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
│   └── BertForRace.py
├── result
│   └── asc01
├── hp_optimize.py
├── train.py
├── README.md
├── requirements.txt
└── LICENSE
```

Please put the data and pre-trained model as above.

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
    - [ ] Add ASC Score to Log
- [ ] Text Logging (Should be **same** as baseline code, override pl original progress bar)
- [ ] Argparse
- [x] Inference & Answer Saving
- [x] Hyper Parameter Tuning (Optuna)
    - [ ] More parameters
- [x] Parallelism
- [ ] Distributed
- [x] ~~Cross Validation~~ (Useless)
- [ ] Data Augmentation
- [ ] Bert Model Tweak
- [ ] Model Ensemble
- [x] ~~Find Best Seed~~ (Useless, there will be new datasets and pre-trained model on-site)
- [ ] Further Speedup Training Process
