"""
SOL-Nav Model: Qwen3-Embedding-0.6B with LoRA + Multi-Step Classification Heads.

Architecture:
1. Pre-trained Qwen3-Embedding-0.6B backbone with LoRA fine-tuning.
2. Mean pooling over the sequence for robust representation.
3. N independent classification heads for multi-step action prediction.
4. Weighted cross-entropy loss to handle action class imbalance.

Key design choices:
- Mean pooling (not CLS): more robust for long structured observation prompts.
- LayerNorm before classification heads: stabilizes training with bf16/fp16.
- Clamped class weights: prevents extreme loss values from rare actions.
- Xavier init for classification heads: ensures stable gradient flow.
"""

import os
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import AutoModel


class SOLNavMultiStepClassifier(nn.Module):
    """SOL-Nav multi-step action classifier.

    Args:
        model_name: HuggingFace model name or path.
        num_labels: number of action classes (default 4).
        class_weights: tensor of shape (num_labels,) for weighted loss.
        num_steps: number of action steps to predict (action chunk size).
        cache_dir: HuggingFace cache directory.
        lora_rank: LoRA rank.
        lora_alpha: LoRA alpha.
        lora_dropout: LoRA dropout rate.
        lora_target_modules: list of module names to apply LoRA to.
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int = 4,
        class_weights: torch.Tensor = None,
        num_steps: int = 4,
        cache_dir: str = None,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: list = None,
    ):
        super().__init__()
        self.num_labels = num_labels
        self.num_steps = num_steps
        self.model_name = model_name

        if lora_target_modules is None:
            lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        # 1. Load base model
        self.base_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
            device_map=None,
        )

        # 2. Apply LoRA
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=None,
        )
        self.base_model = get_peft_model(self.base_model, lora_config)
        self.base_model.print_trainable_parameters()

        self.embedding_dim = self.base_model.config.hidden_size

        # 3. Multi-step prediction heads with LayerNorm for stability
        self.prediction_heads = nn.ModuleList()
        for _ in range(num_steps):
            head = nn.Sequential(
                nn.LayerNorm(self.embedding_dim),
                nn.Linear(self.embedding_dim, self.embedding_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(self.embedding_dim // 2, num_labels),
            )
            # Xavier init for Linear layers
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
            self.prediction_heads.append(head)

        # 4. Class weights (clamped to prevent extreme values)
        if class_weights is not None:
            class_weights_clamped = torch.clamp(class_weights, min=0.1, max=10.0)
            self.register_buffer("class_weights", class_weights_clamped)
        else:
            self.register_buffer("class_weights", torch.ones(num_labels))

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        """Forward pass.

        Args:
            input_ids: (B, seq_len)
            attention_mask: (B, seq_len)
            labels: (B, num_steps) action labels, -100 for ignored positions.

        Returns:
            If labels provided: (loss, logits)
            Otherwise: logits
        """
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Mean pooling (more robust than CLS token for embeddings)
        mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size())
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        sequence_embedding = sum_embeddings / sum_mask

        # Multi-step prediction
        logits_list = []
        for head in self.prediction_heads:
            step_logits = head(sequence_embedding)
            logits_list.append(step_logits)
        logits = torch.stack(logits_list, dim=1)  # (B, num_steps, num_labels)

        # Loss computation
        loss = None
        if labels is not None:
            cw = self.class_weights.to(logits.device)
            total_loss = 0.0
            valid_steps = 0

            for step in range(self.num_steps):
                step_logits = logits[:, step, :]
                step_labels = labels[:, step]

                valid_mask = step_labels != -100
                if not torch.any(valid_mask):
                    continue

                loss_fn = nn.CrossEntropyLoss(weight=cw)
                step_loss = loss_fn(step_logits[valid_mask], step_labels[valid_mask])
                total_loss += step_loss
                valid_steps += 1

            if valid_steps > 0:
                loss = total_loss / valid_steps
            else:
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)

        return (loss, logits) if loss is not None else logits

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the base model."""
        if hasattr(self.base_model, "gradient_checkpointing_enable"):
            self.base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing for the base model."""
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            self.base_model.gradient_checkpointing_disable()

    def save_checkpoint(self, save_dir: str):
        """Save model checkpoint (LoRA weights + config + heads).

        Args:
            save_dir: directory to save checkpoint.
        """
        os.makedirs(save_dir, exist_ok=True)
        # Save LoRA adapter
        self.base_model.save_pretrained(save_dir)
        # Save prediction heads
        torch.save({
            "prediction_heads": self.prediction_heads.state_dict(),
            "class_weights": self.class_weights,
            "num_labels": self.num_labels,
            "num_steps": self.num_steps,
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name,
        }, os.path.join(save_dir, "solnav_heads.pt"))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        cache_dir: str = None,
        device: str = "cpu",
    ):
        """Load model from checkpoint directory.

        Args:
            checkpoint_dir: directory containing saved checkpoint.
            cache_dir: HuggingFace cache directory.
            device: device to load model on.

        Returns:
            SOLNavMultiStepClassifier instance.
        """
        heads_path = os.path.join(checkpoint_dir, "solnav_heads.pt")
        if os.path.exists(heads_path):
            heads_config = torch.load(heads_path, map_location="cpu", weights_only=False)
        else:
            # Fallback to model_config.pt
            config_path = os.path.join(checkpoint_dir, "model_config.pt")
            if os.path.exists(config_path):
                heads_config = torch.load(config_path, map_location="cpu", weights_only=False)
            else:
                heads_config = {}

        model_name = heads_config.get("model_name", checkpoint_dir)
        num_labels = heads_config.get("num_labels", 4)
        num_steps = heads_config.get("num_steps", 4)
        class_weights = heads_config.get("class_weights", None)

        # Create model without LoRA first
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        instance.num_labels = num_labels
        instance.num_steps = num_steps
        instance.model_name = model_name

        # Load base model with LoRA adapter
        # First load the original base model, then apply the adapter
        base_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
        instance.base_model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        instance.embedding_dim = instance.base_model.config.hidden_size

        # Load prediction heads
        instance.prediction_heads = nn.ModuleList()
        embedding_dim = instance.embedding_dim
        for _ in range(num_steps):
            head = nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, embedding_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(embedding_dim // 2, num_labels),
            )
            instance.prediction_heads.append(head)

        if os.path.exists(heads_path) and "prediction_heads" in heads_config:
            instance.prediction_heads.load_state_dict(heads_config["prediction_heads"])

        # Register class weights
        if class_weights is not None:
            class_weights_clamped = torch.clamp(class_weights, min=0.1, max=10.0)
            instance.register_buffer("class_weights", class_weights_clamped)
        else:
            instance.register_buffer("class_weights", torch.ones(num_labels))

        instance = instance.to(device)
        return instance
