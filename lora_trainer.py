import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_linear_schedule_with_warmup
from datasets import load_dataset

class TrainerWithStats:
    def __init__(self, model, tokenizer, train_dataset, cfg):
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
        self.opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=cfg['lr'])
        total_steps = len(self.dataloader) * cfg['epochs']
        self.scheduler = get_linear_schedule_with_warmup(self.opt, 
                                                         num_warmup_steps=cfg['warmup_steps'],
                                                         num_training_steps=total_steps)
        self.device = cfg.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def count_params(self):
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable

    def measure_memory(self):
        if self.device.startswith('cuda'):
            return torch.cuda.max_memory_allocated(self.device) / 1024**2  # MB
        return None

    def train(self, epochs):
        stats = {'step': [], 'loss': [], 
                 'forward_time': [], 'backward_time': [], 'mem_MB': []}
        step = 0
        for epoch in range(epochs):
            for batch in self.dataloader:
                # Подготовка
                inputs = self.tokenizer(batch['text'], return_tensors='pt',
                                        padding=True, truncation=True).to(self.device)
                # Forward
                torch.cuda.reset_peak_memory_stats(self.device)
                t0 = time.perf_counter()
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                t1 = time.perf_counter()
                # Backward
                self.opt.zero_grad()
                loss.backward()
                t2 = time.perf_counter()
                self.opt.step()
                self.scheduler.step()

                # Сбор статистики
                stats['step'].append(step)
                stats['loss'].append(loss.item())
                stats['forward_time'].append(t1 - t0)
                stats['backward_time'].append(t2 - t1)
                stats['mem_MB'].append(self.measure_memory())
                step += 1

        return stats
