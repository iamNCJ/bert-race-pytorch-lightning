# bert-race-pytorch-lightening

bert for RACE with pytorch-lightening and transformer

adopted from [ASC2021-RACE](https://github.com/ASC-Competition/ASC2021-RACE)

## File Structure

```bash
.
├── bert-large-uncased
│   ├── config.json
│   ├── pytorch_model.bin
│   └── vocab.txt
├── data
│   ├── RACE
│   ├── RACEDataModule.py
│   ├── RACELocalLoader.py
│   └── RACELocalLoader.py.lock
├── hp_optimize.py
├── LICENSE
├── model
│   └── BertForRace.py
├── README.md
├── requirements.txt
├── result
│   └── asc01
├── train_offline.py
└── train_online.py
```

Please put the data and pre-trained model as above.

## Environment

```bash
pip install -r requirements.txt
```

You need to install `apex` separately

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
- [ ] Inference & Answer Saving
- [x] Optuna
    - [ ] More parameters
- [ ] Parallelism
- [ ] Distributed
- [ ] Cross Validation
- [ ] Data Augmentation
- [ ] Bert Model Tweak
- [ ] Model Ensemble
- [ ] Find Best Seed