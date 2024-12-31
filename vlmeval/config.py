from vlmeval.vlm import *
from vlmeval.api import *
from functools import partial

ungrouped = {
    # 'MiniCPM-V':partial(MiniCPM_V, model_path='openbmb/MiniCPM-V'),
    # 'MiniCPM-V-2':partial(MiniCPM_V, model_path='openbmb/MiniCPM-V-2'),
    'MiniCPM-Llama3-V-2_5': partial(MiniCPM_Llama3_V, model_path='/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/MiniCPM-Llama3-V-2_5/'),
    'llava_finetune': partial(LLaVA_OneVision, model_path='/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/llava-onevision-qwen2-0.5b-finetune'),
    'llava_finetune_400k': partial(LLaVA_OneVision, model_path='/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/llava-onevision-qwen2-0.5b-finetune_multilingual_400K'),
    'llava_chitb_eng_600k_step_9000': partial(LLaVA_OneVision, model_path='/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/llava-onevision-qwen2-0.5b-finetune_eng_600k/checkpoint-9000'),
    'vai_0.5_finetune': partial(BunnyQwen2, model_path="/raid/phogpt_team/chitb/MultimodalAwesome/save_models/vai_0.5_finetune_main_merge"),
    'vai_0.5_finetune_debug': partial(BunnyQwen2, model_path='/raid/phogpt_team/chitb/MultimodalAwesome/test/vai_0.5_finetune_debug_merge_non_shuffle_with_peft'),
    'vai_0.5_minicpm_demo': partial(BunnyQwen2, model_path='/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/vai_0.5_finetune_main_easy2hard_minicpm_debug_8gpu_merge'),
    'llava_1m_mid_merge': partial(LLaVA_OneVision, model_path='/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/llava_mid_1m_merge_qwen'),
    'vai_0.5_debug_lora_no_short': partial(BunnyQwen2, model_path='/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/vai_qwen_merge_no_short'),
    'vai_0.5_debug_lora_no_short_400k': partial(BunnyQwen2, model_path='/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/vai_qwen_0.5_merge_lora_no_short_400k'),
    'vai_0.5_debug_400k_no_short': partial(BunnyQwen2, model_path='/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/vai_qwen_0.5_merge_lora_400k_no_short'),
    'vai_0.5_debug_400k+text_only': partial(BunnyQwen2, model_path='/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/vai_qwen_0.5_merge_lora_400k+text_only'),
    'vai_0.5+ocr': partial(BunnyQwen2, model_path='/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/vai_qwen_0.5_merge_lora_400k+text_only'),
    'vai_0.5_2epoch': partial(BunnyQwen2, model_path='/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/vai_qwen_0.5_merge_2epoch'),
    'vai_0.5_2epoch_130000_vit': partial(BunnyQwen2, model_path='/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/vai_qwen_0.5_merge_2epoch_130000_vit'),
    'vai_0.5_2epoch_131000_vit': partial(BunnyQwen2, model_path='/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/vai_qwen_0.5_merge_2epoch_131000_vit'),
    'vai_0.5_2epoch_133000_vit': partial(BunnyQwen2, model_path='/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/vai_qwen_0.5_merge_2epoch_133000_vit'),
    'vai_0.5_2epoch_134000_vit': partial(BunnyQwen2, model_path='/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/vai_qwen_0.5_merge_2epoch_134000_vit'),
    'vai_0.5_2epoch_135000_vit': partial(BunnyQwen2, model_path='/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/vai_qwen_0.5_merge_2epoch_135000_vit'),
    'vai_0.5_2epoch_136000_vit': partial(BunnyQwen2, model_path='/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/vai_qwen_0.5_merge_2epoch_136000_vit'),
    'vai_0.5_2epoch_137000_vit': partial(BunnyQwen2, model_path='/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/vai_qwen_0.5_merge_2epoch_137000_vit'),
    'vai_0.5_2epoch_138000_vit': partial(BunnyQwen2, model_path='/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/vai_qwen_0.5_merge_2epoch_138000_vit'),
    'vai_0.5_3epoch': partial(BunnyQwen2, model_path='/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/vai_0.5_merge_qwen_2epoch_continue_3epoch'),
    'vai_0.5_4epoch': partial(BunnyQwen2, model_path='/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/vai_0.5_qwen_merge_2epoch_continue_4epoch/'),
    'vai_0.5_5epoch': partial(BunnyQwen2, model_path='/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/vai_0.5_4epoch_continue_5epoch_merge_qwen/'),
    'vai_0.5_6epoch': partial(BunnyQwen2, model_path='/raid/phogpt_team/chitb/eval/MiniCPM-V/eval_mm/vlmevalkit/vai_0.5_4epoch_continue_6epoch_merge_qwen'),




}


supported_VLM = {}

model_groups = [
    ungrouped
]

for grp in model_groups:
    supported_VLM.update(grp)
