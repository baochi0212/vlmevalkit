import os
import warnings
import torch

from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig, logging
from bunny.model.multimodal_encoder.siglip.siglip_encoder import SiglipVisionModel
logging.set_verbosity_error()
warnings.filterwarnings('ignore')

from bunny.model import *


def load_pretrained_model(model_path, model_base, model_name, model_type, load_8bit=False, load_4bit=False,
                          device_map="auto", device="cuda", **kwargs):
    print("Load from model path: ", model_path)
    if model_type not in {'phi-1.5', 'phi-2', 'phi-3', 'stablelm-2', 'qwen2', 'minicpm', 'llama3-8b'}:
        raise ValueError(f"Unknown Model Type {model_type}")

    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.bfloat16

    # Load Bunny model
    if 'lora' in model_name.lower() and model_base is None:
        warnings.warn(
            'There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
    #load model directly without model base
    if 'merge' in model_path:
        print("Finetune checkpoint")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        model = BunnyQwen2ForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        model = model.cuda()
        print("Model: ", model)
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            # non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            # non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in
            #                        non_lora_trainables.items()}
            # if any(k.startswith('model.model.') for k in non_lora_trainables):
            #     non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in
            #                            non_lora_trainables.items()}
            # model.load_state_dict(non_lora_trainables, strict=False)
            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')

    elif 'lora' in model_name.lower() and model_base is not None:
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        print("Lora checkpoint + model_base")
        print('Loading Bunny from base model...')
        if model_type == 'phi-1.5' or model_type == 'phi-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyPhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                        config=lora_cfg_pretrained, **kwargs)
        elif model_type == 'phi-3':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyPhi3ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                         config=lora_cfg_pretrained, **kwargs)
        elif model_type == 'stablelm-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, trust_remote_code=True)
            model = BunnyStableLMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                             config=lora_cfg_pretrained, **kwargs)
        elif model_type == 'qwen2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained,
                                                          **kwargs)
        elif model_type == 'minicpm':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyMiniCPMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                            config=lora_cfg_pretrained,
                                                            **kwargs)
        elif model_type == 'llama3-8b':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                          config=lora_cfg_pretrained,
                                                          **kwargs)

        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(
                torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(
                torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional Bunny weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        # else:
        #     # this is probably from HF Hub
        #     from huggingface_hub import hf_hub_download
        #     def load_from_hf(repo_id, filename, subfolder=None):
        #         cache_file = hf_hub_download(
        #             repo_id=repo_id,
        #             filename=filename,
        #             subfolder=subfolder)
        #         return torch.load(cache_file, map_location='cpu')

        #     non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')

        non_lora_trainables = {(k[18:] if k.startswith('module.base_model.') else k): v for k, v in
                               non_lora_trainables.items()}
        #print("Non lora keys: ", non_lora_trainables.keys())
        print("Before load lora: ", model)
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in
                                   non_lora_trainables.items()}
        #print("Load non-lora-tranables: ", model.load_state_dict(non_lora_trainables, strict=False))
        print("Non lora keys: ", non_lora_trainables.keys())    
        print("Model after non_lora: ", model)
   


        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        
        print('Merging LoRA weights...', model)
        model = model.merge_and_unload() #no vision tower if delay load 
        print('Model is loaded...')
    elif model_base is not None:
        # this may be mm projector only
        print('Loading Bunny from base model...')

        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        if model_type == 'phi-1.5' or model_type == 'phi-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyPhiForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                        config=cfg_pretrained, **kwargs)
        elif model_type == 'phi-3':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyPhi3ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                         config=cfg_pretrained, **kwargs)
        elif model_type == 'stablelm-2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True, trust_remote_code=True)
            model = BunnyStableLMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True,
                                                             config=cfg_pretrained, **kwargs)
        elif model_type == 'qwen2':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            #print(cfg_pretrained)
            model = BunnyQwen2ForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                                                          **kwargs)
        
        elif model_type == 'minicpm':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyMiniCPMForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                                                            **kwargs)
        elif model_type == 'llama3-8b':
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
            model = BunnyLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained,
                                                          **kwargs)
        print("Config: ", cfg_pretrained)
        mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
        mm_projector_weights = {k: v.to(torch.bfloat16) for k, v in mm_projector_weights.items()}
        model.load_state_dict(mm_projector_weights, strict=False)
    else:
        if model_type == 'phi-1.5' or model_type == 'phi-2':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyPhiForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'phi-3':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyPhi3ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'stablelm-2':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
            model = BunnyStableLMForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'qwen2':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyQwen2ForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'minicpm':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyMiniCPMForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
        elif model_type == 'llama3-8b':
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
            model = BunnyLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
    #RESIZE to llm vocab
    #print("???", len(tokenizer))
    model.resize_token_embeddings(len(tokenizer))
    #get the image_processor

    #if not continous training aka delay_load=False -> vision_tower None (pretrained merged unloaded)
    vision_tower = model.get_vision_tower()
    #print("Before is loaded: ", vision_tower.is_loaded, vision_tower.vision_tower.vision_model.embeddings.position_embedding.weight[0])
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    #print(vision_tower)
    from copy import deepcopy
    
    #debug 
    siglip_model = SiglipVisionModel.from_pretrained("/raid/phogpt_team/chitb/checkpoint_spp/siglip-so400m-patch14-384")
    print(siglip_model.vision_model.embeddings.patch_embedding.weight[0].to(torch.bfloat16),  model.model.mm_projector[0].weight[0])


    #before_weigths = deepcopy(vision_tower.vision_tower.vision_model.embeddings.position_embedding.weight[0])
    #print(before_weigths)
    #LOAD VISION TOWER weight, which can't be loaded from merge (non_lora_trainables.bin & vision_tower.bin)
    #continue finetuning checkpoint already load the vision tower
    #if "merge" not in model_path and "finetune_main" not in model_path:
    if "merge" not in model_path:
        print("Model config", model.config)
        if getattr(model.config, "unfreeze_vision_tower", False):
            print("Load trainable VISION TOWER!!!!!")
            if 'lora' in model_name.lower():
                print("Load from non_lora_trainables.bin")
                assert model_base is not None
                vision_non_lora_trainables = {k[19:]: v for k, v in non_lora_trainables.items() if
                                              k.startswith('model.vision_tower.')}
                projector_non_lora_trainables = {k[19:]: v for k, v in non_lora_trainables.items() if
                                              k.startswith('model.mm_projector.')}
                print("Load non-lora mlp: ", model.model.mm_projector.load_state_dict(projector_non_lora_trainables, strict=False))
                print("Load non-lora vision_tower: ", vision_tower.load_state_dict(vision_non_lora_trainables, strict=False))
            elif mode_base is not None:
                print("Load from vision_tower.bin")
                #assert model_base is None
                # from safetensors.torch import load_file
                # vision_weights = {}
                # for file_name in os.listdir(model_path):
                #     if file_name.endswith('safetensors'):
                #         vision_weights.update(
                #             {k[19:]: v for k, v in load_file(os.path.join(model_path, file_name)).items() if
                #              k.startswith('model.vision_tower.')})
        
                state_dict = torch.load(f"{model_path}/vision_tower.bin")
                vision_tower_weights = {k[19:]: v for k, v in state_dict.items()
                                        if k.startswith('model.vision_tower.')}
                vision_tower.load_state_dict(vision_tower_weights, strict=True)
        else:
            print("Load non-trainable VISION TOWER!!!!!")
    vision_tower.to(device=device, dtype=torch.bfloat16)
    print("After vs Before loading vision tower weights: ", model.model.vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight[0], model.model.mm_projector[0].weight[0])
    image_processor = vision_tower.image_processor
    #model.model.vision_tower = vision_tower
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    if model_type == 'llama3-8b':
        tokenizer.eos_token_id = 128001
        model.generation_config.pad_token_id = tokenizer.eos_token_id

    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = model.generation_config.eos_token_id
    print("Sanity device check: ", model.device, vision_tower.device)
    return tokenizer, model, image_processor, context_len
