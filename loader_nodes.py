# ComfyUI/custom_nodes/Qwen_Image_Toolkit/loader_nodes.py
import torch
import comfy.utils
import comfy.sd
import folder_paths
import os
import json

try:
    from safetensors.torch import safe_open
    print("[Qwen Toolkit] Using standard 'safetensors.torch.safe_open'.")
except ImportError:
    print("[Qwen Toolkit] Warning: Could not import from 'safetensors.torch'. Falling back to 'comfy.utils.safe_open'.")
    from comfy.utils import safe_open

class QwenImageLoraLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_alpha": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 128.0,
                    "step": 0.1,
                    "tooltips": "Set to 0.0 for fully automatic detection (metadata/heuristic). Specify a value to override."
                }),
            }
        }
        
    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_qwen_lora"
    CATEGORY = "loaders/Qwen_Image_Toolkit"

    def _get_lora_alpha(self, lora_path, manual_alpha, lora_state_dict):
        if manual_alpha > 0.0:
            print(f"[Qwen Lora] Using manually provided alpha: {manual_alpha}")
            return manual_alpha

        try:
            with safe_open(lora_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                if metadata:
                    if 'ss_lora_alpha' in metadata:
                        alpha = float(metadata['ss_lora_alpha'])
                        print(f"[Qwen Lora] Found alpha in safetensors metadata (ss_lora_alpha): {alpha}")
                        return alpha
                    if 'lora_alpha' in metadata:
                        alpha = float(metadata['lora_alpha'])
                        print(f"[Qwen Lora] Found alpha in safetensors metadata (lora_alpha): {alpha}")
                        return alpha
        except Exception as e:
            print(f"[Qwen Lora] Note: Could not read safetensors metadata ({e}).")

        try:
            lora_rank = None
            for key, value in lora_state_dict.items():
                if key.endswith((".lora_A.weight", ".lora_A.default.weight", ".lora.down.weight")):
                    lora_rank = value.shape[0]
                    break
            if lora_rank:
                heuristic_alpha = float(lora_rank)
                print(f"[Qwen Lora] Inferred rank {lora_rank}, using heuristic alpha: {heuristic_alpha}")
                return heuristic_alpha
        except Exception as e:
            print(f"[Qwen Lora] Warning: Could not infer rank for heuristic alpha: {e}")
            
        config_path = os.path.splitext(lora_path)[0] + '.json'
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f: config = json.load(f)
                alpha = config.get('lora_alpha')
                if alpha is not None:
                    print(f"[Qwen Lora] Found alpha in config file: {alpha}")
                    return float(alpha)
            except Exception as e:
                print(f"[Qwen Lora] Warning: Could not read or parse config.json: {e}")

        default_alpha = 16.0
        print(f"[Qwen Lora] Could not determine alpha automatically. Falling back to default: {default_alpha}")
        return default_alpha

    def _convert_keys(self, state_dict, lora_alpha):
        new_sd = {}
        is_libre_format = any(k.startswith("transformer.") for k in state_dict)
        
        if is_libre_format:
            print("[Qwen Lora] Detected 'Libre' format. Applying conversion rules.")
        else:
            print("[Qwen Lora] Detected 'Motang' or similar format. Applying conversion rules.")
            
        for key, value in state_dict.items():
            if 'lora_' not in key: continue
            
            processed_key = None
            temp_key = key

            if is_libre_format:
                if temp_key.startswith("transformer."):
                    temp_key = temp_key.replace("transformer.", "", 1)
                
                if temp_key.endswith(".lora_A.weight"):
                    processed_key = temp_key.replace(".lora_A.weight", ".lora.down.weight")
                elif temp_key.endswith(".lora_B.weight"):
                    processed_key = temp_key.replace(".lora_B.weight", ".lora.up.weight")
            else:
                temp_key = temp_key.replace(".default.weight", "")
                if temp_key.endswith(".lora_A"):
                    processed_key = temp_key.replace(".lora_A", ".lora.down.weight")
                elif temp_key.endswith(".lora_B"):
                    processed_key = temp_key.replace(".lora_B", ".lora.up.weight")

            if processed_key:
                final_key = "diffusion_model." + processed_key
                new_sd[final_key] = value
                
                if final_key.endswith(".lora.down.weight"):
                    alpha_key = final_key.replace(".lora.down.weight", ".alpha")
                    new_sd[alpha_key] = torch.tensor(float(lora_alpha))

        return new_sd

    def load_qwen_lora(self, model, clip, lora_name, strength_model, strength_clip, lora_alpha):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)
            
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora_state_dict = comfy.utils.load_torch_file(lora_path, safe_load=True)
        
        alpha = self._get_lora_alpha(lora_path, lora_alpha, lora_state_dict)
        
        print(f"[Qwen Lora] Converting LoRA '{lora_name}' with final alpha={alpha}...")
        converted_lora_state_dict = self._convert_keys(lora_state_dict, alpha)
        
        if not converted_lora_state_dict:
            print(f"[Qwen Lora] FATAL: Key conversion resulted in an empty dictionary. LoRA format is unrecognized. LoRA will not be applied.")
            return (model, clip)
            
        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, converted_lora_state_dict, strength_model, strength_clip)
        
        print(f"[Qwen Lora] Successfully applied LoRA '{lora_name}'.")
        return (model_lora, clip_lora)
