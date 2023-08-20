# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, ConcatenateIterator
from transformers import AutoModel, AutoTokenizer

import patch_chat_glm


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            "./chatglm2-6b", trust_remote_code=True, local_files_only=True
        )
        model = AutoModel.from_pretrained(
            "./chatglm2-6b", trust_remote_code=True, local_files_only=True
        ).cuda()
        patch_chat_glm.patch(model)
        self.model = model.eval()

    def predict(
        self,
        prompt: str = Input(
            description="Prompt for completion",
            default="[Round 1]\n\n问：请使用英文重复这段话：\"为了使模型生成最优输出，当使用 ChatGLM2-6B 时需要使用特定的输入格式，请按照示例格式组织输入。\"\n\n答：",
        ),
        max_tokens: int = Input(
            description="Max new tokens to generate", default=2048, ge=1, le=32768
        ),
        temperature: float = Input(description="Temperature", default=0.75, ge=0, le=5),
        top_p: float = Input(description="Top_p", default=0.8, ge=0, le=1),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""
        
        yield from self.model.stream_completion(
            self.tokenizer, prompt, max_new_tokens=max_tokens, temperature=temperature, top_p=top_p
        )
