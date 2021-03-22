# bert-race-pytorch-lightening
 bert for RACE with pytorch-lightening and transformer

 adopted from [ASC2021-RACE](https://github.com/ASC-Competition/ASC2021-RACE)

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
 - [ ] Text Logging (Should be **same** as baseline code)
 - [ ] Argparse
 - [ ] Inference & Answer Saving
 - [x] Optuna
   - [ ] More parameters
 - [ ] Parallelism
 - [ ] Distributed
 - [ ] Data Augmentation
 - [ ] Bert Model Tweak
 - [ ] Model Ensemble