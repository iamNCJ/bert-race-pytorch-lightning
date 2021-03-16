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
 - [ ] Refactor RACE Dataset Loader
   - [x] Use `datasets` from `transformer`
   - [x] Better Interface and Format
   - [x] Faster Data Loading (using Rust)
   - [x] Cache Tokenized Results
   - [ ] Custom Datasets
 - [ ] Apex
 - [ ] Optuna
 - [ ] Parallelism
 - [ ] Distributed
 - [ ] Data Augmentation
 - [ ] Bert Model Tweak
 - [ ] Model Ensemble