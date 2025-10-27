import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse
import math
import time
import os
import numpy as np

from gemlite.helper import *

from transformers import AttentionInterface
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

device = 'cuda:0'

model_id = 'meta-llama/Llama-3.2-3B-Instruct'

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation='flash_attention_2',
    torch_dtype=torch.bfloat16
).to(device)
model.eval()

import torch
from torch.profiler import profile, ProfilerActivity

tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)


inputs = tok("hello", return_tensors="pt").to(model.device)
with torch.no_grad(), profile(activities=[ProfilerActivity.CUDA]) as prof:
    _ = model(**inputs)
print([k.key for k in prof.key_averages()])
print([k.key for k in prof.key_averages() if "flash_attn" in k.key.lower()])

