import types

import torch

from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


@torch.no_grad()
def completion(
    self,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 8192,
    num_beams=1,
    do_sample=True,
    top_p=0.8,
    temperature=0.8,
    logits_processor=None,
    **kwargs
):
    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "do_sample": do_sample,
        "top_p": top_p,
        "temperature": temperature,
        "logits_processor": logits_processor,
        **kwargs,
    }
    inputs = tokenizer([prompt], return_tensors="pt").to(self.device)
    outputs = self.generate(**inputs, **gen_kwargs)
    outputs = outputs.tolist()[0][len(inputs["input_ids"][0]) :]
    response = tokenizer.decode(outputs)
    return response


@torch.no_grad()
def stream_completion(
    self,
    tokenizer,
    prompt: str,
    past_key_values=None,
    max_new_tokens: int = 8192,
    do_sample=True,
    top_p=0.8,
    temperature=0.8,
    logits_processor=None,
    return_past_key_values=False,
    **kwargs
):
    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "top_p": top_p,
        "temperature": temperature,
        "logits_processor": logits_processor,
        **kwargs,
    }
    if past_key_values is None and not return_past_key_values:
        inputs = tokenizer([prompt], return_tensors="pt").to(self.device)
    else:
        input_ids = tokenizer.encode("\n\n" + prompt, add_special_tokens=False)
        input_ids = input_ids[1:]
        inputs = tokenizer.batch_encode_plus(
            [(input_ids, None)], return_tensors="pt", add_special_tokens=False
        ).to(self.device)
    if past_key_values is not None:
        past_length = past_key_values[0][0].shape[0]
        inputs.position_ids += past_length
        attention_mask = inputs.attention_mask
        attention_mask = torch.cat(
            (attention_mask.new_ones(1, past_length), attention_mask), dim=1
        )
        inputs["attention_mask"] = attention_mask
    offset = 0
    for outputs in self.stream_generate(
        **inputs,
        past_key_values=past_key_values,
        return_past_key_values=return_past_key_values,
        **gen_kwargs
    ):
        if return_past_key_values:
            outputs, past_key_values = outputs
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]) :]
        response = tokenizer.decode(outputs)
        if response and response[-1] != "�":
            if return_past_key_values:
                yield response[offset:], past_key_values
            else:
                yield response[offset:]
        offset = len(response)


def patch(model):
    model.stream_completion = types.MethodType(stream_completion, model)
    model.completion = types.MethodType(completion, model)
