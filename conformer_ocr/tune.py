#!/usr/bin/env python
import click

from conformer_ocr.cli.util import _expand_gt, _validate_manifests, message

RECOGNITION_HYPER_PARAMS = {'pad': 16,
                            'freq': 1.0,
                            'batch_size': 1,
                            'quit': 'fixed',
                            'epochs': 30,
                            'min_epochs': 30,
                            'lag': 10,
                            'min_delta': None,
                            'optimizer': 'Adam',
                            'lrate': 1e-3,
                            'momentum': 0.9,
                            'weight_decay': 0.0,
                            'schedule': 'constant',
                            'normalization': 'NFD',
                            'normalize_whitespace': True,
                            'completed_epochs': 0,
                            'augment': False,
                            # lr scheduler params
                            # step/exp decay
                            'step_size': 10,
                            'gamma': 0.1,
                            # reduce on plateau
                            'rop_factor': 0.1,
                            'rop_patience': 5,
                            # cosine
                            'cos_t_max': 50,
                            'warmup': 0,
                            'freeze_backbone': 0,
                            'height': 96,
                            'encoder_dim': 512,
                            'num_encoder_layers': 17,
                            'num_attention_heads': 8,
                            'feed_forward_expansion_factor':4,
                            'conv_expansion_factor': 2,
                            'input_dropout_p': 0.1,
                            'feed_forward_dropout_p': 0.1,
                            'attention_dropout_p':0.1,
                            'conv_dropout_p':0.1,
                            'conv_kernel_size': 31,
                            'half_step_residual': True,
                            'decoder_hidden_dim': 512,
                            }

def train_model(config, format_type, training_data, evaluation_data):

    from conformer_ocr.dataset import TextLineDataModule
    from conformer_ocr.model import RecognitionModel

    from pytorch_lightning import Trainer
    from threadpoolctl import threadpool_limits


    hyper_params = RECOGNITION_HYPER_PARAMS.copy()
    hyper_params.update(config)

    data_module = TextLineDataModule(training_data=training_data,
                                     evaluation_data=evaluation_data,
                                     pad=hyper_params['pad'],
                                     height=hyper_params['height'],
                                     augmentation=hyper_params['augment'],
                                     partition=0.9,
                                     batch_size=hyper_params['batch_size'],
                                     num_workers=8,
                                     format_type=format_type)

    model = RecognitionModel(hyper_params=hyper_params,
                             num_classes=data_module.num_classes,
                             batches_per_epoch=len(data_module.train_dataloader()))

    trainer = Trainer(accelerator="auto",
                      devices="auto",
                      precision=16,
                      max_epochs=hyper_params['epochs'],
                      min_epochs=hyper_params['min_epochs'],
                      enable_progress_bar=False,
                      enable_model_summary=False,
                      enable_checkpointing=False,
                      callbacks=[RayTrainReportCallback()],
                      plugins=[RayLightningEnvironment()])

    trainer = prepare_trainer(trainer)
    with threadpool_limits(limits=threads):
        trainer.fit(model, data_module)


@click.command()
@click.version_option()
@click.pass_context
@click.option('-s', '--seed', default=None, type=click.INT,
              help='Seed for numpy\'s and torch\'s RNG. Set to a fixed value to '
                   'ensure reproducible random splits of data')
@click.option('-o', '--output', show_default=True, type=click.Path(), default='/mnt/nfs_data/experiments/cocr', help='Output directory for checkpoints')
@click.option('-N', '--epochs', show_default=True, default=RECOGNITION_HYPER_PARAMS['epochs'], help='Number of epochs to train for')
@click.option('-S', '--samples', show_default=True, default=25, help='Number of samples')
@click.option('-w', '--workers', show_default=True, default=3, help='Number of ray tune workers')
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
def cli(ctx, seed, output, epochs, samples, workers, training_files,
        evaluation_files, format_type, ground_truth):

    from functools import partial

    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    from ray.train.lightning import (
        RayLightningEnvironment,
        RayTrainReportCallback,
        prepare_trainer,
    )
    from ray.train import ScalingConfig
    from ray.train.torch import TorchTrainer
    from ray.train import RunConfig, ScalingConfig, CheckpointConfig

    if seed:
        from pytorch_lightning import seed_everything
        seed_everything(seed, workers=True)

    search_space = {
        "warmup": tune.lograndint(1, 10000),
        "lr": tune.loguniform(1e-6, 1e-1),
        "batch_size": tune.choice([1, 2, 4, 8, 16]),
        'decoder_hidden_dim': tune.lograndint(128, 512),
    }


    # disable automatic partition when given evaluation set explicitly
    if evaluation_files:
        partition = 1
    ground_truth = list(ground_truth)

    # merge training_files into ground_truth list
    if training_files:
        ground_truth.extend(training_files)

    if len(ground_truth) == 0:
        raise click.UsageError('No training data was provided to the train command. Use `-t` or the `ground_truth` argument.')

    train_cocr = partial(train_model, format_type=format_type, training_data=ground_truth, evaluation_data=evaluation_files)

    scaling_config = ScalingConfig(
        num_workers=workers, use_gpu=True, resources_per_worker={"CPU": 8, "GPU": 1}
    )

    run_config = RunConfig(
        checkpoint_config=CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute="val_metric",
            checkpoint_score_order="max",
        ),
        storage_path=output,
        name="cocr_tune",
    )

    # Define a TorchTrainer without hyper-parameters for Tuner
    ray_trainer = TorchTrainer(
        train_cocr,
        scaling_config=scaling_config,
        run_config=run_config,
    )

    def tune_cocr_asha(num_samples=25, num_epochs=50):
        scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

        tuner = tune.Tuner(
            ray_trainer,
            param_space={"train_loop_config": search_space},
            tune_config=tune.TuneConfig(
                metric="val_metric",
                mode="max",
                num_samples=num_samples,
                scheduler=scheduler,
            ),
        )
        return tuner.fit()

    results = tune_cocr_asha(num_samples=samples, num_epochs=epochs)

if __name__ == '__main__':
    cli()
