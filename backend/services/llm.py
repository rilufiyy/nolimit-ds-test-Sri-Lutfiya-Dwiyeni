"""
Shared LLM service — supports both seq2seq (T5/BART) and causal (Qwen2, etc.) models.

Default: Qwen/Qwen2-0.5B-Instruct (~370 MB fp16, instruction-tuned, conversational)
Fallback seq2seq: google/flan-t5-base (override via RAG_LLM_MODEL env var)

Auto-detects model architecture at load time:
  - Causal (decoder-only) : Qwen2, LLaMA, Mistral, Phi, GPT-2, etc.
  - Seq2Seq (enc-decoder) : T5, BART, mT5, etc.
"""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

LLM_MODEL_ID: str = os.getenv("RAG_LLM_MODEL", "Qwen/Qwen2-0.5B-Instruct")
_MAX_INPUT_TOKENS: int = 1024

# model_type values that indicate a decoder-only causal LM
_CAUSAL_FAMILIES = {
    "llama", "qwen2", "mistral", "phi", "phi-msft", "gpt2", "gpt_neo",
    "gpt_neox", "bloom", "opt", "falcon", "stablelm", "internlm",
    "baichuan", "chatglm", "gemma", "gemma2",
}


class LLMService:
    """Singleton LLM for text generation — auto-detects seq2seq vs causal."""

    _tokenizer = None
    _model = None
    _is_causal: bool = False

    def _load(self) -> None:
        if LLMService._tokenizer is not None:
            return
        from transformers import (
            AutoConfig,
            AutoModelForCausalLM,
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
        )

        logger.info(f"Loading LLM: {LLM_MODEL_ID}")
        config = AutoConfig.from_pretrained(LLM_MODEL_ID)
        model_type = config.model_type.lower()
        is_causal = model_type in _CAUSAL_FAMILIES

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        LLMService._tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)

        if is_causal:
            LLMService._model = AutoModelForCausalLM.from_pretrained(
                LLM_MODEL_ID, torch_dtype=dtype,
            )
        else:
            LLMService._model = AutoModelForSeq2SeqLM.from_pretrained(
                LLM_MODEL_ID, torch_dtype=dtype,
            )

        if torch.cuda.is_available():
            LLMService._model = LLMService._model.cuda()
        LLMService._model.eval()
        LLMService._is_causal = is_causal
        logger.info(f"LLM ready: {LLM_MODEL_ID} (causal={is_causal})")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        num_beams: int = 1,
    ) -> str:
        """Run generation and return decoded string."""
        self._load()
        tok = LLMService._tokenizer
        model = LLMService._model

        if LLMService._is_causal:
            # Build chat-formatted input using the model's own chat template
            if tok.chat_template:
                messages = [{"role": "user", "content": prompt}]
                input_text = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                input_text = prompt

            inputs = tok(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=_MAX_INPUT_TOKENS,
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            input_len = inputs["input_ids"].shape[1]
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                )
            # Decode only the newly generated tokens (skip the echoed prompt)
            new_ids = output_ids[0][input_len:]
            return tok.decode(new_ids, skip_special_tokens=True).strip()

        else:
            # Seq2seq path (T5, BART, etc.)
            inputs = tok(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=_MAX_INPUT_TOKENS,
            )
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    early_stopping=True,
                )
            return tok.decode(output_ids[0], skip_special_tokens=True).strip()


llm_service = LLMService()