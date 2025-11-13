import math
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from lora import LoRAInjector

class LoRATrainer:
    """
    • Встраивает LoRA в модель
    • Считает trainable / total параметры
    • Запускает цикл обучения (FW+BW)
    • Возвращает лосс / perplexity
    """
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer,
        dataset,            # HuggingFace Dataset, уже с полями input_ids, attention_mask
        device: str = "cuda",
        lora_cfg: dict = None,
        train_cfg: dict = None
    ):
        self.device    = device
        self.tokenizer = tokenizer
        self.dataset   = dataset

        # LoRA-инъекция
        
        injector = LoRAInjector(model, **(lora_cfg or {}))
        injector.add_lora()
        self.model = injector.model.to(device)

        # оптимизируем только LoRA-параметры
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            **(train_cfg.get("optim", {}) if train_cfg else {})
        )

        # DataLoader
        bs = train_cfg.get("batch_size", 8) if train_cfg else 8
        self.dl = DataLoader(dataset, batch_size=bs, shuffle=True)

        # Гиперы обучения
        self.epochs    = train_cfg.get("epochs", 3) if train_cfg else 3
        self.max_len   = train_cfg.get("max_length", 128) if train_cfg else 128

    def count_parameters(self) -> dict:
        total = sum(p.numel() for p in self.model.parameters())
        train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"total": total, "trainable": train}

    def train(self):
        """Запускает обучение; возвращает список (loss, ppl) по эпохам."""
        self.model.train()
        results = []
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for batch in self.dl:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                labels = batch["input_ids"]
                out = self.model(**batch, labels=labels)
                loss = out.loss

                loss.backward()
                self.optimizer.step(); self.optimizer.zero_grad()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.dl)
            ppl = math.exp(avg_loss)
            results.append((avg_loss, ppl))
            print(f"Epoch {epoch}: loss={avg_loss:.4f}, ppl={ppl:.2f}")
        return results
