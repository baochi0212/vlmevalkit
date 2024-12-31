import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import warnings
import re
from mmawe.constants import (
    BUNNY_IMAGE_TOKEN_INDEX, BUNNY_DEFAULT_IMAGE_TOKEN
)
from mmawe.conversation import conv_templates, SeparatorStyle
from .bunny_builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from mmawe.util.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from PIL import Image
from io import BytesIO
from .base import BaseModel
from ..smp import *
from ..utils import DATASET_TYPE


class BunnyQwen2(BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = False

    def __init__(self, model_path='BAAI/Bunny-v1_1-Llama-3-8B-V', **kwargs):
        assert model_path is not None
        transformers.logging.set_verbosity_error()
        transformers.logging.disable_progress_bar()
        warnings.filterwarnings('ignore')
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path, None, 'bunny_lora', model_type='qwen2'
        )
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto', trust_remote_code=True)
        self.tokenizer, self.model, self.image_processor = tokenizer, model, image_processor
        self.kwargs = kwargs

    def use_custom_prompt(self, dataset):
        if listinstr(['MCQ', 'Y/N'], DATASET_TYPE(dataset)) or listinstr(['mathvista'], dataset.lower()):
            return True
        else:
            return False

    def build_prompt(self, line, dataset):
        if dataset is None:
            dataset = self.dataset

        if isinstance(line, int):
            line = self.data.iloc[line]

        tgt_path = self.dump_image(line, dataset)

        prompt = line['question']

        if DATASET_TYPE(dataset) == 'MCQ':
            if listinstr(['mmmu'], dataset.lower()):
                hint = line['hint'] if (
                    'hint' in line and not pd.isna(line['hint'])) else None
                assert hint is None

                question = line['question']
                question = re.sub(r'<image (\d+)>',
                                  lambda x: x.group(0)[1:-1], question)

                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                options_prompt = '\n'
                for key, item in options.items():
                    options_prompt += f'({key}) {item}\n'

                prompt = question
                if len(options):
                    prompt += options_prompt
                    prompt += "\nAnswer with the option's letter from the given choices directly."
                else:
                    prompt += '\nAnswer the question using a single word or phrase.'
            else:
                hint = line['hint'] if (
                    'hint' in line and not pd.isna(line['hint'])) else None
                prompt = ''
                if hint is not None:
                    prompt += f'{hint}\n'

                question = line['question']

                options = {
                    cand: line[cand]
                    for cand in string.ascii_uppercase
                    if cand in line and not pd.isna(line[cand])
                }
                options_prompt = '\n'
                for key, item in options.items():
                    options_prompt += f'{key}. {item}\n'

                prompt += question + options_prompt
                if listinstr(['cn', 'ccbench'], dataset.lower()):
                    prompt += '请直接回答选项字母。'
                else:
                    prompt += "Answer with the option's letter from the given choices directly."
        elif DATASET_TYPE(dataset) == 'Y/N':
            if listinstr(['mme'], dataset.lower()):
                if not listinstr(
                        ['code_reasoning', 'commonsense_reasoning',
                            'numerical_calculation', 'text_translation'],
                        line['category']):
                    prompt = prompt.replace(' Please answer yes or no.',
                                            '\nAnswer the question using a single word or phrase.')
            elif listinstr(['pope'], dataset.lower()):
                prompt = prompt.replace(' Please answer yes or no.',
                                        '\nAnswer the question using a single word or phrase.')
        elif listinstr(['mathvista'], dataset.lower()):
            match = re.search(
                r'Hint: (.*?)\nQuestion: (.*?)\n(Choices:\n(.*))?', prompt + '\n', re.DOTALL)

            prompt = match.group(2)
            if match.group(4) is not None:
                prompt += '\n' + match.group(4).rstrip('\n')
            prompt += '\n' + match.group(1)
        else:
            raise ValueError(
                f"Bunny doesn't implement a custom prompt for {dataset}. It should use the default prompt, but didn't.")

        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))

        return msgs

    def generate_inner(self, message, dataset=None, *args, **kwargs):

        prompt, image_path = self.message_to_promptimg(message)

        # text = (f'A chat between a curious user and an artificial intelligence assistant. '
        #         f"The assistant gives helpful, detailed, and polite answers to the user's questions. "
        #         f'USER: <image>\n{prompt} ASSISTANT:')

        # text_chunks = [self.tokenizer(chunk).input_ids for chunk in text.split('<image>')]
        # input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long).unsqueeze(0)
        prompt = BUNNY_DEFAULT_IMAGE_TOKEN + '\n' + prompt
        # text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{prompt} ASSISTANT:"
        conv = conv_templates["qwen2"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        # text_chunks = [tokenizer(chunk).input_ids for chunk in text.split('<image>')]
        # input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long).unsqueeze(0).to(device)
        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer,
                                  BUNNY_IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .cuda()
        )
        image = Image.open(image_path).convert('RGB')
        image_tensor = process_images([image],
                                      self.image_processor,
                                      self.model.config).to(dtype=self.model.dtype)

        output_ids = self.model.generate(input_ids,
                                         images=image_tensor,
                                         do_sample=False,
                                         max_new_tokens=256,
                                         use_cache=True,
                                         )[0]
        response = self.tokenizer.decode(
            output_ids[input_ids.shape[1]:], skip_special_tokens=True)
        return response
