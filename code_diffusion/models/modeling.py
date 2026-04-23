from __future__ import annotations

import contextlib
import inspect
import types
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from code_diffusion.utils.tokenization import ensure_padding_token


def resolve_device(device_name: str) -> str:
    if device_name != "auto":
        return device_name
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def _build_attention_bias(
    attention_mask: torch.Tensor | None,
    input_ids: torch.Tensor,
    *,
    mode: str,
    mask_positions: torch.Tensor | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    min_value = torch.finfo(dtype).min

    if attention_mask is None:
        token_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)
    else:
        token_mask = attention_mask.to(device=device, dtype=torch.bool)

    allow = token_mask[:, :, None] & token_mask[:, None, :]

    if mode == "conditioned" and mask_positions is not None:
        masked_queries = mask_positions.to(device=device, dtype=torch.bool)[:, :, None]
        non_masked_keys = (~mask_positions.to(device=device, dtype=torch.bool))[:, None, :]
        allow = torch.where(masked_queries, allow & non_masked_keys, allow)

    # Avoid rows with all -inf, which can produce NaNs on some backends.
    valid_queries = token_mask[:, :, None]
    allow = torch.where(valid_queries, allow, token_mask[:, None, :].expand_as(allow))

    bias = torch.zeros((batch_size, 1, seq_len, seq_len), dtype=dtype, device=device)
    bias.masked_fill_(~allow[:, None, :, :], min_value)
    return bias


@contextlib.contextmanager
def _patched_causal_mask(base_model: nn.Module | None, attention_bias: torch.Tensor):
    if base_model is None or not hasattr(base_model, "_update_causal_mask"):
        yield False
        return

    original = base_model._update_causal_mask

    def _patched(self, *args, **kwargs):
        return attention_bias

    base_model._update_causal_mask = types.MethodType(_patched, base_model)
    try:
        yield True
    finally:
        base_model._update_causal_mask = original


class DiffusionCodeModel(nn.Module):
    def __init__(self, model: nn.Module, tokenizer, attention_mode: str = "full") -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.attention_mode = attention_mode
        self.base_model = getattr(model, getattr(model, "base_model_prefix", "model"), None)
        self.has_native_bidirectional = hasattr(model.config, "use_bidirectional_attention")

    def forward(
        self,
        input_ids: torch.LongTensor,
        *,
        attention_mask: torch.Tensor | None = None,
        mask_positions: torch.Tensor | None = None,
        use_cache: bool = False,
    ):
        if self.attention_mode == "full" and self.has_native_bidirectional:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=use_cache,
            )

        attention_bias = _build_attention_bias(
            attention_mask=attention_mask,
            input_ids=input_ids,
            mode=self.attention_mode,
            mask_positions=mask_positions,
            dtype=next(self.model.parameters()).dtype,
        )

        with _patched_causal_mask(self.base_model, attention_bias) as patched:
            if patched:
                return self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=use_cache,
                )

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_bias,
            use_cache=use_cache,
        )

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(output_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)


def load_diffusion_model(config: dict) -> tuple[DiffusionCodeModel, object]:
    model_name = config["model_name"]
    resolved_device = resolve_device(config.get("device", "auto"))
    trust_remote_code = bool(config.get("trust_remote_code", False))
    adapter_config_path = Path(model_name) / "adapter_config.json"
    adapter_checkpoint = adapter_config_path.exists()
    quantized_loading = config.get("finetune_method") == "qlora" and resolved_device.startswith("cuda")

    base_model_name = model_name
    tokenizer_source = model_name
    if adapter_checkpoint:
        from peft import PeftConfig

        peft_config = PeftConfig.from_pretrained(model_name)
        base_model_name = peft_config.base_model_name_or_path
        if not _checkpoint_has_tokenizer_files(Path(model_name)):
            tokenizer_source = base_model_name

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=trust_remote_code)
    ensure_padding_token(tokenizer)

    model_config = AutoConfig.from_pretrained(base_model_name, trust_remote_code=trust_remote_code)
    if config.get("attention_mode", "full") == "full" and hasattr(
        model_config, "use_bidirectional_attention"
    ):
        model_config.use_bidirectional_attention = _resolve_bidirectional_mode(
            model_config=model_config,
            requested_value=config.get("bidirectional_attention_value"),
        )

    torch_dtype = resolve_dtype(config.get("dtype", "float32"))
    from_pretrained_kwargs = {
        "config": model_config,
        "trust_remote_code": trust_remote_code,
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    if config.get("attn_implementation"):
        from_pretrained_kwargs["attn_implementation"] = config["attn_implementation"]
    if quantized_loading:
        from_pretrained_kwargs["quantization_config"] = _build_quantization_config(config)
        from_pretrained_kwargs["device_map"] = _resolve_quantized_device_map(resolved_device)

    model = AutoModelForCausalLM.from_pretrained(base_model_name, **from_pretrained_kwargs)
    model.config.use_cache = False

    if adapter_checkpoint:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, model_name, is_trainable=False)
    else:
        model = _apply_finetuning_strategy(model, config)

    wrapper = DiffusionCodeModel(
        model=model,
        tokenizer=tokenizer,
        attention_mode=config.get("attention_mode", "full"),
    )

    if not quantized_loading and not adapter_checkpoint:
        wrapper.to(resolved_device)
    elif adapter_checkpoint and resolved_device != "cpu" and not quantized_loading:
        wrapper.to(resolved_device)

    _log_trainable_parameter_summary(wrapper.model)
    return wrapper, tokenizer


def _resolve_bidirectional_mode(model_config, requested_value):
    if requested_value is not None:
        return requested_value
    if getattr(model_config, "model_type", "") == "gemma4":
        return "all"
    return True


def _build_quantization_config(config: dict):
    from transformers import BitsAndBytesConfig

    compute_dtype = resolve_dtype(config.get("qlora_compute_dtype", config.get("dtype", "bfloat16")))
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=config.get("qlora_quant_type", "nf4"),
        bnb_4bit_use_double_quant=bool(config.get("qlora_use_double_quant", True)),
    )


def _apply_finetuning_strategy(model: nn.Module, config: dict) -> nn.Module:
    finetune_method = config.get("finetune_method", "full")
    if finetune_method == "full":
        _maybe_enable_gradient_checkpointing(model, config)
        return model

    from peft import LoraConfig, TaskType, get_peft_model

    if finetune_method == "qlora":
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(config.get("gradient_checkpointing", True)),
        )
    else:
        _maybe_enable_gradient_checkpointing(model, config)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(config.get("lora_r", 32)),
        lora_alpha=int(config.get("lora_alpha", 64)),
        lora_dropout=float(config.get("lora_dropout", 0.05)),
        bias=config.get("lora_bias", "none"),
        target_modules=config.get("lora_target_modules", "all-linear"),
    )
    model = get_peft_model(model, lora_config)
    return model


def _maybe_enable_gradient_checkpointing(model: nn.Module, config: dict) -> None:
    if not config.get("gradient_checkpointing", True):
        return
    if hasattr(model, "gradient_checkpointing_enable"):
        enable = model.gradient_checkpointing_enable
        signature = inspect.signature(enable)
        if "gradient_checkpointing_kwargs" in signature.parameters:
            enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False


def _resolve_quantized_device_map(device: str):
    if device == "cuda":
        return {"": 0}
    if device.startswith("cuda:"):
        return {"": int(device.split(":", 1)[1])}
    return {"": device}


def _checkpoint_has_tokenizer_files(path: Path) -> bool:
    return any(
        (path / filename).exists()
        for filename in (
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "spiece.model",
        )
    )


def _log_trainable_parameter_summary(model: nn.Module) -> None:
    trainable = 0
    total = 0
    for parameter in model.parameters():
        count = parameter.numel()
        total += count
        if parameter.requires_grad:
            trainable += count
    ratio = 100.0 * trainable / max(total, 1)
    print(
        f"trainable_parameters={trainable} "
        f"total_parameters={total} "
        f"trainable_ratio={ratio:.4f}%"
    )
