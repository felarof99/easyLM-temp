#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().system('pip install ipywidgets -q --upgrade')
# get_ipython().system('pip install torch --index-url https://download.pytorch.org/whl/cpu -q --upgrade')
# get_ipython().system('pip install --upgrade jax -q  --upgrade')
# get_ipython().system('pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html -q --upgrade')
# get_ipython().system('pip install "flax[all]" -q --upgrade')
# get_ipython().system('pip install --upgrade optax==0.2.2')
# get_ipython().system('pip install --upgrade einops')
# get_ipython().system('pip install --no-cache-dir transformers==4.43.3')
# get_ipython().system('pip install --no-cache-dir datasets==2.18.0')
# get_ipython().system('pip install --upgrade tqdm')
# get_ipython().system('pip install --upgrade requests')
# get_ipython().system('pip install --upgrade typing-extensions')
# get_ipython().system('pip install --upgrade mlxu>=0.1.13')
# get_ipython().system('pip install --upgrade sentencepiece')
# get_ipython().system('pip install --upgrade pydantic')
# get_ipython().system('pip install --upgrade fastapi')
# get_ipython().system('pip install --upgrade uvicorn')
# get_ipython().system('pip install --upgrade gradio')

HF_DIR = '/home/felarof99/data/hf/'

import os
os.environ['HF_HUB_CACHE'] = HF_DIR
os.environ['HF_HOME'] = HF_DIR
get_ipython().system('export HF_HUB_CACHE=HF_DIR')
get_ipython().system('export HF_HOME=HF_DIR')

os.makedirs(HF_DIR, exist_ok=True)


import os
import sys
import importlib
import sys
import os
from types import SimpleNamespace
def import_local_module(module_path: str):
    sys.path.append('')
    module = importlib.import_module(module_path)
    return importlib.reload(module)

convert_hf_to_easylm = import_local_module("EasyLM.models.llama.convert_hf_to_easylm")

llama_model = import_local_module("EasyLM.models.llama.llama_model")
output_file = "/home/felarof99/data/easy/llama3.1_405b.flax"

os.makedirs(os.path.dirname(output_file), exist_ok=True)

args = SimpleNamespace(
    hf_model="meta-llama/Meta-Llama-3.1-405B-Instruct",
    output_file=output_file,
    streaming=True,
    float_dtype="bf16"
)

convert_hf_to_easylm.FLAGS = args

# Set up the llama configuration
convert_hf_to_easylm.FLAGS.llama = llama_model.LLaMAConfigurator.get_default_config()
convert_hf_to_easylm.FLAGS.llama.base_model = "llama3_8b"

convert_hf_to_easylm.main([])


from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="/home/felarof99/data/easy/",
    repo_id="felafax/llama-3.1-405B-Instruct-JAX",
    repo_type="model",
    ignore_patterns=[".*"],
    token="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"
)

