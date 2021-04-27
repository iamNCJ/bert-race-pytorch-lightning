import optuna
import pytorch_lightning as pl
import torch
from optuna.integration import PyTorchLightningPruningCallback

from data.RACEDataModule import RACEDataModule
from model.BertForRace import BertForRace


def objective(trial: optuna.trial.Trial) -> float:
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 2e-5)

    model = BertForRace(
        pretrained_model='./model/bert-large-uncased',
        learning_rate=learning_rate,
        num_train_epochs=20,
        train_batch_size=4,
        train_all=True,
    )
    dm = RACEDataModule(
        model_name_or_path='./model/bert-large-uncased',
        datasets_loader='./data/RACELocalLoader.py',
        train_batch_size=32,
        max_seq_length=128,
        num_workers=8,
        num_preprocess_processes=48,
        use_sentence_selection=True,
    )
    trainer = pl.Trainer(
        gpus=-1 if torch.cuda.is_available() else None,
        # callbacks=[checkpoint_callback],
        amp_backend='native',
        amp_level='O2',
        precision=16,
        accelerator='ddp',
        gradient_clip_val=1.0,
        max_epochs=3,
        plugins='ddp_sharded',
        callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
        # limit_train_batches=0.1,
        # limit_val_batches=0.1,
        # accumulate_grad_batches=2,
    )
    trainer.fit(model, dm)
    return trainer.callback_metrics["val_acc"].item()


if __name__ == '__main__':
    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner()
    )

    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=100, timeout=600)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: {}".format(best_trial.value))

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))
