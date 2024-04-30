#
# Copyright 2024 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
Default hyperparameters
"""

RECOGNITION_HYPER_PARAMS = {'pad': 16,
                            'freq': 1.0,
                            'batch_size': 32,
                            'quit': 'fixed',
                            'epochs': 100,
                            'min_epochs': 10,
                            'lag': 10,
                            'min_delta': None,
                            'optimizer': 'AdamW',
                            'lr': 3e-4,
                            'momentum': 0.9,
                            'weight_decay': 1e-5,
                            'schedule': 'cosine',
                            'normalization': 'NFD',
                            'normalize_whitespace': True,
                            'completed_epochs': 0,
                            'augment': True,
                            # lr scheduler params
                            # step/exp decay
                            'step_size': 10,
                            'gamma': 0.1,
                            # reduce on plateau
                            'rop_factor': 0.1,
                            'rop_patience': 5,
                            # cosine
                            'cos_t_max': 100,
                            'cos_min_lr': 3e-5,
                            'warmup': 35000,
                            'freeze_backbone': 0,
                            'height': 96,
                            'encoder_dim': 144,
                            'num_encoder_layers': 16,
                            'num_attention_heads': 4,
                            'feed_forward_expansion_factor': 4,
                            'conv_expansion_factor': 2,
                            'input_dropout_p': 0.1,
                            'feed_forward_dropout_p': 0.1,
                            'attention_dropout_p': 0.1,
                            'conv_dropout_p': 0.1,
                            'conv_kernel_size': 31,
                            'half_step_residual': True,
                            'subsampling_conv_channels': 32,
                            'subsampling_factor': 4,
                            }
