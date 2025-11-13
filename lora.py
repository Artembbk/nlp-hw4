# lora.py
import torch
import torch.nn as nn
import time

class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: int = 8,
        dropout: float = 0.0,
        merge_on_eval: bool = True
    ):
        super().__init__()
        assert r > 0, "rank r must be > 0"
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        self.merge_on_eval = merge_on_eval

        self.weight = nn.Parameter(base_layer.weight.data, requires_grad=False)
        self.has_bias = base_layer.bias is not None
        if self.has_bias:
            self.bias = nn.Parameter(base_layer.bias.data, requires_grad=False)

        in_f, out_f = base_layer.in_features, base_layer.out_features

        self.drop = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        self.A = nn.Parameter(torch.randn(r, in_f) * 0.02)
        self.B = nn.Parameter(torch.zeros(out_f, r))

        self.to(self.weight.dtype)

    @torch.no_grad()
    def merge(self) -> None:
        delta_w = self.B @ self.A
        self.weight += delta_w * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            Ax = self.drop(torch.matmul(x, self.A.t()))
            BAx = torch.matmul(Ax, self.B.t()) * self.scale
            out = torch.matmul(x, self.weight.t()) + BAx
        else:
            if self.merge_on_eval and not hasattr(self, "_merged"):
                self.merge(); self._merged = True
            out = torch.matmul(x, self.weight.t())
        if self.has_bias:
            out = out + self.bias
        return out

class LoRAInjector:
    def __init__(self, model: nn.Module, r=4, alpha=8, dropout=0.):
        """
        model   – ваша исходная модель
        cfg     – конфигурация LoRA
        added   – список кортежей (path, old_module, new_module)
        """
        self.model, self.cfg = model, dict(r=r, alpha=alpha, dropout=dropout)
        self.added = []                      # (module_path, old_module, new_module)

        # сразу заморозим все параметры
        for p in self.model.parameters():
            p.requires_grad = False

    # ----------------- вставка LoRA -----------------
    def _replace(self, parent, name, old):
        """
        Заменяем линейный слой old → LoRALinear, 
        сохраняем в added для дальнейшей настройки requires_grad
        """
        new = LoRALinear(old, **self.cfg)
        setattr(parent, name, new)
        self.added.append((f"{parent}.{name}", old, new))

        # у новых LoRA-параметров включаем градиент
        for p in new.lora_A.parameters():
            p.requires_grad = True
        for p in new.lora_B.parameters():
            p.requires_grad = True
        # если у LoRALinear есть bias или alpha, их тоже разморозим:
        if hasattr(new, "bias") and new.bias is not None:
            new.bias.requires_grad = True
        if hasattr(new, "lora_alpha"):
            new.lora_alpha.requires_grad = True

    def add_lora(self, target_names=("q_proj", "v_proj")):
        """
        Рекурсивно ищем nn.Linear по именам target_names,
        заменяем их на LoRALinear, при этом:
          - все исходные параметры изначально заморожены
          - активируем градиент только для новых лор-слоёв
        """
        def _walk(mod: nn.Module):
            for n, child in list(mod.named_children()):
                if isinstance(child, nn.Linear) and n in target_names:
                    print("insert")
                    self._replace(mod, n, child)
                else:
                    _walk(child)

        _walk(self.model)

        # опционально: выводим информацию о вставленных слоях
        print(f"Inserted LoRA into {len(self.added)} modules:")
        for path, old, new in self.added:
            print(f"  • {path}: {old} → {new}")