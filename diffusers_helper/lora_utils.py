from pathlib import Path
import re
from typing import Dict, List, Optional, Union
from diffusers.loaders.lora_pipeline import _fetch_state_dict
from diffusers.loaders.lora_conversion_utils import _convert_hunyuan_video_lora_to_diffusers
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers.loaders.peft import _SET_ADAPTER_SCALE_FN_MAPPING
import torch

def load_lora(transformer, lora_path: Path, weight_name: Optional[str] = "pytorch_lora_weights.safetensors"):
    """
    Load LoRA weights into the transformer model.

    Args:
        transformer: The transformer model to which LoRA weights will be applied.
        lora_path (Path): Path to the LoRA weights file.
        weight_name (Optional[str]): Name of the weight to load.

    """
    
    state_dict = _fetch_state_dict(
        lora_path,
        weight_name,
        True,
        True,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None)

    state_dict = _convert_hunyuan_video_lora_to_diffusers(state_dict)

    adapter_name = weight_name.split(".")[0]

    # Check if adapter already exists and delete it if it does
    if hasattr(transformer, 'peft_config') and adapter_name in transformer.peft_config:
        print(f"Adapter '{adapter_name}' already exists. Removing it before loading again.")
        # Use delete_adapters (plural) instead of delete_adapter
        transformer.delete_adapters([adapter_name])

    # Add a safety check for empty state_dict
    if not state_dict:
        print(f"Warning: Empty state dict for LoRA '{adapter_name}', skipping.")
        return transformer

    # Add safety check for rank values
    rank_pattern = re.compile(r'.*\.lora_A\.weight')
    rank_values = []
    for key in state_dict.keys():
        if rank_pattern.match(key):
            rank_values.append(state_dict[key].shape[0])

    if not rank_values:
        print(f"Warning: No rank values found in LoRA '{adapter_name}', using default rank 4")
        rank = 4
        # Create dummy rank dict for peft
        for key in list(state_dict.keys()):
            if key.endswith(".weight"):
                base_key = key.rsplit(".", 1)[0]
                if "lora_A" in base_key:
                    dummy_key = base_key.replace("lora_A", "lora_B")
                    if dummy_key + ".weight" not in state_dict:
                        print(f"Adding dummy rank for {dummy_key}")
                        state_dict[f"{base_key}.alpha"] = torch.tensor(rank)

    try:
        # Load the adapter with the original name
        transformer.load_lora_adapter(state_dict, network_alphas=None, adapter_name=adapter_name)
        print(f"LoRA weights '{adapter_name}' loaded successfully.")
    except Exception as e:
        print(f"Error loading LoRA '{adapter_name}': {str(e)}")
        print("This may be due to an incompatible LoRA format. Skipping.")
    
    return transformer

def unload_all_loras(transformer):
    """
    Unload all LoRA adapters from the transformer model.

    Args:
        transformer: The transformer model from which to unload LoRA adapters.

    Returns:
        The transformer model with all LoRA adapters unloaded.
    """
    if transformer is None:
        print("Transformer is None, cannot unload LoRAs")
        return transformer

    if hasattr(transformer, 'peft_config') and transformer.peft_config:
        adapter_names = list(transformer.peft_config.keys())
        if adapter_names:
            print(f"Unloading all LoRAs: {', '.join(adapter_names)}")
            transformer.delete_adapters(adapter_names)
            
            # Force cleanup of any remaining adapter references
            if hasattr(transformer, 'active_adapter'):
                transformer.active_adapter = None
                
            # Clear any cached states
            for module in transformer.modules():
                if hasattr(module, 'lora_A'):
                    if isinstance(module.lora_A, dict):
                        module.lora_A.clear()
                if hasattr(module, 'lora_B'):
                    if isinstance(module.lora_B, dict):
                        module.lora_B.clear()
                if hasattr(module, 'scaling'):
                    if isinstance(module.scaling, dict):
                        module.scaling.clear()
            
            print("All LoRAs unloaded successfully")
    else:
        print("No LoRAs loaded in transformer")
    
    # Force garbage collection
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return transformer

# TODO(neph1): remove when HunyuanVideoTransformer3DModelPacked is in _SET_ADAPTER_SCALE_FN_MAPPING
def set_adapters(
        transformer,
        adapter_names: Union[List[str], str],
        weights: Optional[Union[float, Dict, List[float], List[Dict], List[None]]] = None,
    ):

    adapter_names = [adapter_names] if isinstance(adapter_names, str) else adapter_names

    # Expand weights into a list, one entry per adapter
    # examples for e.g. 2 adapters:  [{...}, 7] -> [7,7] ; None -> [None, None]
    if not isinstance(weights, list):
        weights = [weights] * len(adapter_names)

    if len(adapter_names) != len(weights):
        raise ValueError(
            f"Length of adapter names {len(adapter_names)} is not equal to the length of their weights {len(weights)}."
        )

    # Set None values to default of 1.0
    # e.g. [{...}, 7] -> [{...}, 7] ; [None, None] -> [1.0, 1.0]
    weights = [w if w is not None else 1.0 for w in weights]

    # e.g. [{...}, 7] -> [{expanded dict...}, 7]
    scale_expansion_fn = _SET_ADAPTER_SCALE_FN_MAPPING["HunyuanVideoTransformer3DModel"]
    weights = scale_expansion_fn(transformer, weights)

    set_weights_and_activate_adapters(transformer, adapter_names, weights)