import optuna
import pytorch_lightning as pl
import torch
from optuna.integration import PyTorchLightningPruningCallback

from data.RACEDataModule import RACEDataModule
from model.BertForRace import BertForRace


def objective(trial: optuna.trial.Trial) -> float:
    learning_rate = trial.suggest_float("learning_rate", 0.001, 2e-5)

    model = BertForRace(learning_rate=learning_rate)
    dm = RACEDataModule(train_batch_size=64)
    trainer = pl.Trainer(gpus=-1 if torch.cuda.is_available() else None,
                         amp_level='O2',
                         precision=16,
                         gradient_clip_val=1.0,
                         max_epochs=1,
                         callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
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
