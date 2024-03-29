#!/usr/bin/env python
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

from conformer_ocr.dataset import TextLineDataModule
from conformer_ocr.model import RecognitionModel

from pytorch_lightning import Trainer


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
                            'normalization': None,
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

search_space = {
    "warmup": tune.lograndint(1, 10000),
    "lr": tune.loguniform(1e-6, 1e-1),
    "batch_size": tune.choice([1, 2, 4, 8, 16]),
}

num_epochs = 30
num_samples = 25

def train_model(config):
    hyper_params = RECOGNITION_HYPER_PARAMS.copy()
    hyper_params.update(config)

    data_module = TextLineDataModule(training_data=ground_truth,
                                     evaluation_data=evaluation_files,
                                     pad=hyper_params['pad'],
                                     height=hyper_params['height'],
                                     augmentation=augment,
                                     partition=partition,
                                     batch_size=batch_size,
                                     num_workers=workers,
                                     reorder=reorder,
                                     binary_dataset_split=fixed_splits,
                                     format_type=format_type,
                                     codec=codec)

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


scaling_config = ScalingConfig(
    num_workers=3, use_gpu=True, resources_per_worker={"CPU": 8, "GPU": 1}
)

run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="val_metric",
        checkpoint_score_order="max",
    ),
    storage_path="/mnt/nfs_data/experiments/cocr",
    name="cocr_tune",
)

# Define a TorchTrainer without hyper-parameters for Tuner
ray_trainer = TorchTrainer(
    train_func,
    scaling_config=scaling_config,
    run_config=run_config,
)

def tune_cocr_asha(num_samples=25):
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

results = tune_cocr_asha(num_samples=num_samples)
