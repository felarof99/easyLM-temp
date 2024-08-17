"""
Usage:
python convert_hf_to_easylm.py  \
       --hf_model     /path/hf_format_dir    \
       --output_file /path/easylm_format.easylm   \
       --llama.base_model llama_7b \
       --streaming
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''
import time
from pathlib import Path

import mlxu
import torch
import flax
from transformers import AutoModelForCausalLM

from EasyLM.models.llama.llama_model import LLaMAConfigurator
from EasyLM.checkpoint import StreamingCheckpointer
from EasyLM.jax_utils import get_float_dtype_by_name


FLAGS, FLAGS_DEF = mlxu.define_flags_with_default(
    # hf_model="",
    # output_file="",
    # streaming=True,
    # float_dtype="bf16",
    # llama=LLaMAConfigurator.get_default_config(),
)


def inverse_permute(w, n_heads, input_dim, output_dim):
    """
    Rearrange the weight matrix for LLaMA's attention mechanism.

    This function performs the inverse of the permutation applied in the original LLaMA implementation.
    It's used to convert weights from HuggingFace format to EasyLM format.

    Args:
        w (numpy.ndarray): Input weight matrix
        n_heads (int): Number of attention heads
        input_dim (int): Input dimension
        output_dim (int): Output dimension

    Returns:
        numpy.ndarray: Rearranged weight matrix
    """
    # Reshape the input weight matrix
    reshaped_w = w.reshape(n_heads, 2, output_dim // n_heads // 2, input_dim)
    
    # Transpose the reshaped matrix to swap dimensions
    transposed_w = reshaped_w.transpose(0, 2, 1, 3)
    
    # Flatten the transposed matrix back to 2D
    inverted_w = transposed_w.reshape(output_dim, input_dim)
    
    return inverted_w


def main(argv):
    start = time.time()
    # Initialize LLaMA configuration
    llama_config = LLaMAConfigurator.finalize_config(FLAGS.llama)
    
    # Request HuggingFace token from user
    huggingface_token = input("INPUT: Please provide your HUGGINGFACE_TOKEN: ")
    
    # Load the HuggingFace model
    hf_model = AutoModelForCausalLM.from_pretrained(FLAGS.hf_model, token=huggingface_token)
    ckpt = hf_model.state_dict()

    print(f"Start convert weight to easylm format...")
    
    # Convert weights to EasyLM format
    jax_weights = {
        "transformer": {
            # Embedding layer
            "wte": {"embedding": ckpt["model.embed_tokens.weight"].numpy()},
            # Final layer norm
            "ln_f": {"kernel": ckpt["model.norm.weight"].numpy()},
            # Transformer layers
            "h": {
                "%d" % (layer): {
                    "attention": {
                        # Query projection
                        "wq": {
                            "kernel": inverse_permute(
                                ckpt[f"model.layers.{layer}.self_attn.q_proj.weight"].numpy(),
                                llama_config.num_attention_heads,
                                llama_config.hidden_size,
                                llama_config.hidden_size,
                            ).transpose()
                        },
                        # Key projection
                        "wk": {
                            "kernel": inverse_permute(
                                ckpt[f"model.layers.{layer}.self_attn.k_proj.weight"].numpy(),
                                llama_config.num_key_value_heads,
                                llama_config.hidden_size,
                                llama_config.hidden_size // (
                                    llama_config.num_attention_heads
                                    // llama_config.num_key_value_heads
                                ),
                            ).transpose()
                        },
                        # Value projection
                        "wv": {
                            "kernel": ckpt[f"model.layers.{layer}.self_attn.v_proj.weight"]
                            .numpy().transpose()
                        },
                        # Output projection
                        "wo": {
                            "kernel": ckpt[f"model.layers.{layer}.self_attn.o_proj.weight"]
                            .numpy().transpose()
                        },
                    },
                    "feed_forward": {
                        # First feed-forward layer
                        "w1": {
                            "kernel": ckpt[f"model.layers.{layer}.mlp.gate_proj.weight"]
                            .numpy().transpose()
                        },
                        # Second feed-forward layer
                        "w2": {
                            "kernel": ckpt[f"model.layers.{layer}.mlp.down_proj.weight"]
                            .numpy().transpose()
                        },
                        # Third feed-forward layer
                        "w3": {
                            "kernel": ckpt[f"model.layers.{layer}.mlp.up_proj.weight"]
                            .numpy().transpose()
                        },
                    },
                    # Layer norm before attention
                    "attention_norm": {
                        "kernel": ckpt[f"model.layers.{layer}.input_layernorm.weight"].numpy()
                    },
                    # Layer norm after attention
                    "ffn_norm": {
                        "kernel": ckpt[
                            f"model.layers.{layer}.post_attention_layernorm.weight"
                        ].numpy()
                    },
                }
                for layer in range(llama_config.num_hidden_layers)
            },
        },
        # Language model head
        "lm_head": {"kernel": ckpt["lm_head.weight"].numpy().transpose()},
    }
    print(f"Convert weight to easylm format finished...")
    print(f"Start to save...")

    # Save weights
    if FLAGS.streaming:
        StreamingCheckpointer.save_train_state_to_file(
            jax_weights,
            FLAGS.output_file,
            float_dtype=get_float_dtype_by_name(FLAGS.float_dtype),
        )
    else:
        with mlxu.open_file(FLAGS.output_file, "wb") as fout:
            fout.write(flax.serialization.msgpack_serialize(jax_weights, in_place=True))

    print(
        f"Save finished!!! take time: {time.time() - start} save path: {FLAGS.output_file}"
    )


if __name__ == "__main__":
    mlxu.run(main)
