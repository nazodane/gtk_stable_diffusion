# Copyright 2022 Toshimitsu Kimura <lovesyao@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from gtk_stable_diffusion.free_weights import free_weights
import time

if True:
    sta_time = time.perf_counter()
    with torch.no_grad():
        from diffusers import DPMSolverMultistepScheduler
        from transformers import AutoFeatureExtractor, BertTokenizerFast, CLIPTextModel, CLIPTokenizer
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
        from gtk_stable_diffusion.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
        import diffusers
        scheduler = DPMSolverMultistepScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
        )

        unet_config = {'sample_size': 32, 'in_channels': 4, 'out_channels': 4, \
                       'down_block_types': ('CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'CrossAttnDownBlock2D', 'DownBlock2D'), \
                       'up_block_types': ('UpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D', 'CrossAttnUpBlock2D'), \
                       'block_out_channels': (320, 640, 1280, 1280), 'layers_per_block': 2, 'cross_attention_dim': 768, 'attention_head_dim': 8}

        unet = diffusers.UNet2DConditionModel(**unet_config)

        vae_config = {'sample_size': 256, 'in_channels': 3, 'out_channels': 3, \
                      'down_block_types': ('DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D', 'DownEncoderBlock2D'), \
                      'up_block_types': ('UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D', 'UpDecoderBlock2D'), \
                      'block_out_channels': (128, 256, 512, 512), 'latent_channels': 4, 'layers_per_block': 2}

        vae = diffusers.AutoencoderKL(**vae_config)

        text_model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        free_weights(unet, vae, text_model)
        end_time = time.perf_counter()
        print("init: %s"%(end_time-sta_time))

        torch.save((unet, vae, text_model), "./gtk_stable_diffusion/ckpt_base.pt")

        sta_time = time.perf_counter()
        (unet, vae, text_model) = torch.load("./gtk_stable_diffusion/ckpt_base.pt")
        end_time = time.perf_counter()
        print("load: %s"%(end_time-sta_time))
