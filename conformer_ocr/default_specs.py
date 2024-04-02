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
                            'batch_size': 1,
                            'quit': 'early',
                            'epochs': -1,
                            'min_epochs': 0,
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
                            'encoder_dim': 256,
                            'num_encoder_layers': 17,
                            'num_attention_heads': 8,
                            'feed_forward_expansion_factor':4,
                            'conv_expansion_factor': 2,
                            'input_dropout_p': 0.1,
                            'feed_forward_dropout_p': 0.1,
                            'attention_dropout_p':0.1,
                            'conv_dropout_p':0.1,
                            'conv_kernel_size': 9,
                            'half_step_residual': True,
                            'subsampling_factor': 8,
                            }
