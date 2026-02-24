import argparse
import os
import random
import torch
import numpy as np
from utils.stereo import stereo, attack_stereo
from utils.utils import StableDiffuser
import torch.nn.functional as F
from utils.utils import *
from utils.dataset import TextualInversionDataset
from utils.apg import *
from torch.utils.data import DataLoader
import copy
import PIL
import os
import random
import torch
import numpy as np
from PIL import Image
from packaging import version
from torch.utils.data import Dataset
from torchvision import transforms
from safetensors.torch import save_file, load_file

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline)


file_path = "/data2/nessessence/bright_workspace/robust-concept-erasing_mine/stereo_weights_mine/David_Beckham/unet_weight_suma_final.safetensors"
pipeline = DiffusionPipeline.from_pretrained(
                'CompVis/stable-diffusion-v1-4',
            ).to("cuda")
pipeline.unet.load_state_dict(load_file(file_path), strict=False)
# pipeline("a photo of David Beckham").images[0].save("test.png")
pipeline("a photo of Barack Obama").images[0].save("test2.png")

