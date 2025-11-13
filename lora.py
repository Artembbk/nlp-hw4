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
        self.model, self.cfg = model, dict(r=r, alpha=alpha, dropout=dropout)
        self.added = []                      # (module_path, old_module, new_module)

    # ----------------- вставка LoRA -----------------
    def _replace(self, parent, name, old):
        new = LoRALinear(old, **self.cfg)
        setattr(parent, name, new)
        self.added.append((f"{parent}.{name}", old, new))

    def add_lora(self, target_names=("q_proj", "v_proj")):
        def _walk(mod: nn.Module):
            for n, child in list(mod.named_children()):
                if isinstance(child, nn.Linear) and n in target_names:
                    self._replace(mod, n, child)
                else:
                    _walk(child)
        _walk(self.model)

    # ----------------- профилирование -----------------
    @torch.no_grad()
    def _fw(self, **batch):
        return self.model(**batch)

    def profile(self, batch, warmup=10, reps=50, device="cuda"):
        batch = {k: v.to(device) for k, v in batch.items()}
        torch.cuda.reset_peak_memory_stats(device)
        self.model.to(device).eval()

        # ── измеряем FW ────────────────────────────────
        for _ in range(warmup): self._fw(**batch)
        torch.cuda.synchronize();  t0 = time.time()
        for _ in range(reps): self._fw(**batch)
        torch.cuda.synchronize();  fw_t = (time.time()-t0)/reps

        # ── измеряем BW ────────────────────────────────
        self.model.train();  loss = None
        for _ in range(warmup):
            out = self._fw(**batch);  loss = out.logits.mean();  loss.backward();  self.model.zero_grad()
        torch.cuda.synchronize();  t0 = time.time()
        for _ in range(reps):
            out = self._fw(**batch);  loss = out.logits.mean();  loss.backward();  self.model.zero_grad()
        torch.cuda.synchronize();  bw_t = (time.time()-t0)/reps

        mem = torch.cuda.max_memory_allocated(device)/1024**2
        return dict(fw_ms=fw_t*1e3, bw_ms=bw_t*1e3, peak_mem_MB=mem)