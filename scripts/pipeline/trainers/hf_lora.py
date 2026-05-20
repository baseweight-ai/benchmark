"""LoRA SFT via pure HuggingFace peft + TRL (stub).

This is a placeholder adapter that documents the contract for an Unsloth-free
backend. The seam between orchestrator and trainer is proven by the existing
`UnslothLoRATrainer`; this stub keeps the registry honest by reserving the
`hf-lora` slot and pointing implementers at the exact set of library calls
the implementation should make:

  1. Model loading
       from transformers import AutoModelForCausalLM, AutoTokenizer
       from transformers import BitsAndBytesConfig
       quant = BitsAndBytesConfig(load_in_4bit=hw_cfg.load_in_4bit,
                                  bnb_4bit_compute_dtype=hw_cfg.load_dtype,
                                  bnb_4bit_quant_type="nf4")
       model = AutoModelForCausalLM.from_pretrained(
           model_cfg.model_id, quantization_config=quant,
           device_map=hw_cfg.device, attn_implementation="flash_attention_2")
       tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_id)

  2. LoRA wiring
       from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
       model = prepare_model_for_kbit_training(model)
       lora_cfg = LoraConfig(
           r=hw_cfg.lora_rank, lora_alpha=hw_cfg.lora_alpha,
           lora_dropout=model_cfg.lora.get("dropout", 0.05),
           bias="none", task_type="CAUSAL_LM",
           target_modules="all-linear", use_rslora=model_cfg.lora.get("use_rslora", True))
       model = get_peft_model(model, lora_cfg)

  3. Completion-only masking (TRL native — replaces Unsloth's train_on_responses_only)
       from trl import DataCollatorForCompletionOnlyLM
       collator = DataCollatorForCompletionOnlyLM(
           response_template=model_cfg.response_part,
           instruction_template=model_cfg.instruction_part,
           tokenizer=tokenizer)
       # Pass `data_collator=collator` to SFTTrainer; do NOT call
       # train_on_responses_only.

  4. Trainer + train loop are identical to the Unsloth path: `SFTTrainer` +
     `SFTConfig` + `EarlyStoppingCallback`, same callbacks (the
     orchestrator-owned `CheckpointCallback` is framework-agnostic), same
     `trainer.train()` + `model.save_pretrained()`.

  5. Reuse the framework-agnostic helpers in `pipeline.trainers.base`:
       - `verify_completion_masking(trainer.train_dataset, echo)` after
         construction (catches mis-matched chat-template markers),
       - `analyze_training(...)` on the loss curves to build diagnostics,
       - `eval_save_steps(...)` for cadence.

When implementing, follow the patterns in `unsloth_lora.py`; that adapter's
post-train extraction (history, hyperparams, dtypes, cleanup) is reusable
verbatim.
"""
from __future__ import annotations

from pipeline.trainers.base import TrainResult, TrainSpec, Trainer, register_trainer


@register_trainer("hf-lora")
class HFLoRATrainer(Trainer):
    """Pure HF + peft + TRL backend (not yet implemented).

    The Unsloth backend is the supported fast path; this stub reserves the
    registry slot so a config can switch backends without changing the
    orchestrator. Implementers should follow the recipe in the module
    docstring.
    """

    name = "hf-lora"

    def train(self, spec: TrainSpec) -> TrainResult:  # pragma: no cover - not implemented
        raise NotImplementedError(
            "HFLoRATrainer is a stub. The Unsloth backend (trainer_id='unsloth-lora') "
            "is the supported path; switch model_cfg.trainer_id to use it. "
            "See pipeline/trainers/hf_lora.py for the implementation recipe."
        )
