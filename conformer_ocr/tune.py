#!/usr/bin/env python
import click
import uuid

from conformer_ocr.cli.util import _expand_gt, _validate_manifests, message, to_ptl_device
from conformer_ocr.default_specs import RECOGNITION_HYPER_PARAMS

def train_model(trial: 'optuna.trial.Trial',
                accelerator,
                device,
                num_workers,
                format_type,
                training_data,
                evaluation_data,
                epochs) -> float:

    from conformer_ocr.dataset import TextLineDataModule
    from conformer_ocr.model import RecognitionModel
    from conformer_ocr.lightning_utils import PyTorchLightningPruningCallback

    from pytorch_lightning import Trainer
    from threadpoolctl import threadpool_limits

    hyper_params = RECOGNITION_HYPER_PARAMS.copy()
    hyper_params['epochs'] = epochs
    hyper_params['cos_t_max'] = hyper_params['cos_t_max']
    hyper_params['batch_size'] = 32

    hyper_params['warmup'] = trial.suggest_int('warmup', 1, 40000, log=True)
#    hyper_params['height'] = trial.suggest_int('height', 48, 128)
    hyper_params['lr'] = trial.suggest_loguniform('lr', 1e-8, 1e-3)
    hyper_params['cos_min_lr'] = hyper_params['lr']/10
    hyper_params['weight_decay'] = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)
    hyper_params['subsampling_factor'] = trial.suggest_categorical('subsampling_factor', [2, 4, 8])
    hyper_params['encoder_dim'] = trial.suggest_categorical('encoder_dim', [144, 256, 320])


    data_module = TextLineDataModule(training_data=training_data,
                                     evaluation_data=evaluation_data,
                                     pad=hyper_params['pad'],
                                     height=hyper_params['height'],
                                     augmentation=hyper_params['augment'],
                                     partition=0.9,
                                     batch_size=hyper_params['batch_size'],
                                     num_workers=num_workers,
                                     format_type=format_type)

    model = RecognitionModel(**hyper_params,
                             num_classes=data_module.num_classes,
                             batches_per_epoch=len(data_module.train_dataloader()))

    trainer = Trainer(accelerator=accelerator,
                      devices=device,
                      precision='bf16',
                      max_epochs=hyper_params['epochs'],
                      min_epochs=hyper_params['min_epochs'],
                      enable_progress_bar=True,
                      enable_model_summary=True,
                      enable_checkpointing=False,
                      callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_metric")])

    with threadpool_limits(limits=1):
        trainer.fit(model, data_module)

    return trainer.callback_metrics["val_metric"].item()


@click.command()
@click.version_option()
@click.pass_context
@click.option('-d', '--device', default='cpu', show_default=True,
              help='Select device to use (cpu, cuda:0, cuda:1, ...)')
@click.option('-s', '--seed', default=None, type=click.INT,
              help='Seed for numpy\'s and torch\'s RNG. Set to a fixed value to '
                   'ensure reproducible random splits of data')
@click.option('--database', show_default=True, default='sqlite:///cocr.db', help='optuna SQL database location')
@click.option('-n', '--name', show_default=True, default=str(uuid.uuid4()), help='trial identifier')
@click.option('-N', '--epochs', show_default=True, default=RECOGNITION_HYPER_PARAMS['epochs'], help='Number of epochs to train for')
@click.option('-S', '--samples', show_default=True, default=25, help='Number of samples')
@click.option('-w', '--workers', show_default=True, default=32, help='Number of dataloader workers')
@click.option('--pruning/--no-pruning', show_default=True, default=True, help='Enables/disables trial pruning')
@click.option('-t', '--training-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with additional paths to training data')
@click.option('-e', '--evaluation-files', show_default=True, default=None, multiple=True,
              callback=_validate_manifests, type=click.File(mode='r', lazy=True),
              help='File(s) with paths to evaluation data. Overrides the `-p` parameter')
@click.option('-f', '--format-type', type=click.Choice(['path', 'xml', 'alto', 'page', 'binary']), default='path',
              help='Sets the training data format. In ALTO and PageXML mode all '
              'data is extracted from xml files containing both line definitions and a '
              'link to source images. In `path` mode arguments are image files '
              'sharing a prefix up to the last extension with `.gt.txt` text files '
              'containing the transcription. In binary mode files are datasets '
              'files containing pre-extracted text lines.')
@click.argument('ground_truth', nargs=-1, callback=_expand_gt, type=click.Path(exists=False, dir_okay=False))
def cli(ctx, device, seed, database, name, epochs, samples, workers, pruning,
        training_files, evaluation_files, format_type, ground_truth):

    try:
        accelerator, device = to_ptl_device(device)
    except Exception as e:
        raise click.BadOptionUsage('device', str(e))

    from functools import partial

    import torch
    import optuna
    from optuna.trial import TrialState

    torch.set_float32_matmul_precision('medium')

    if seed:
        from pytorch_lightning import seed_everything
        seed_everything(seed, workers=True)

    # disable automatic partition when given evaluation set explicitly
    if evaluation_files:
        partition = 1
    ground_truth = list(ground_truth)

    # merge training_files into ground_truth list
    if training_files:
        ground_truth.extend(training_files)

    if len(ground_truth) == 0:
        raise click.UsageError('No training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    objective = partial(train_model,
                        accelerator=accelerator,
                        device=device,
                        format_type=format_type,
                        training_data=ground_truth,
                        evaluation_data=evaluation_files,
                        num_workers=workers,
                        epochs=epochs)

    pruner = optuna.pruners.MedianPruner() if pruning else optuna.pruners.NopPruner()

    print(f'database: {database} trial: {name}')

    study = optuna.create_study(direction="maximize",
                                pruner=pruner,
                                study_name=name,
                                storage=database,
                                load_if_exists=True)
    study.optimize(objective, n_trials=samples)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial



if __name__ == '__main__':
    cli()
