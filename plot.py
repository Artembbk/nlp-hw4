import matplotlib.pyplot as plt

def plot_training_stats(stats, title_suffix=""):
    """
    Рисует графики метрик из словаря stats, возвращённого TrainerWithStats.train().
    
    Аргументы:
        stats (dict): {
            'step': [...],
            'loss': [...],
            'forward_time': [...],
            'backward_time': [...],
            'mem_MB': [...]
        }
        title_suffix (str): суффикс для заголовков графиков
    """
    steps = stats['step']
    loss = stats['loss']
    fwd = stats['forward_time']
    bwd = stats['backward_time']
    mem = stats['mem_MB']
    
    plt.figure(figsize=(12, 10))
    
    # 1. Потери
    plt.subplot(2, 2, 1)
    plt.plot(steps, loss, label="loss", color='tab:red')
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Training Loss {title_suffix}")
    plt.grid(True)
    
    # 2. Время forward
    plt.subplot(2, 2, 2)
    plt.plot(steps, fwd, label="forward_time", color='tab:blue')
    plt.xlabel("Step")
    plt.ylabel("Seconds")
    plt.title(f"Forward Time per Step {title_suffix}")
    plt.grid(True)
    
    # 3. Время backward
    plt.subplot(2, 2, 3)
    plt.plot(steps, bwd, label="backward_time", color='tab:green')
    plt.xlabel("Step")
    plt.ylabel("Seconds")
    plt.title(f"Backward Time per Step {title_suffix}")
    plt.grid(True)
    
    # 4. Потребление памяти
    plt.subplot(2, 2, 4)
    plt.plot(steps, mem, label="mem_MB", color='tab:purple')
    plt.xlabel("Step")
    plt.ylabel("MB")
    plt.title(f"Peak GPU Memory {title_suffix}")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()