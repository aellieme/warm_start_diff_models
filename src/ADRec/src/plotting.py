import matplotlib.pyplot as plt
import os
from collections import defaultdict

class TrainingPlotter:
    def __init__(self, save_dir, model_name='model', metrics=None, figsize=(10, 6)):
        """
        Args:
            save_dir (str): папка для сохранения графиков
            model_name (str)
            metrics (list): список метрик для отслеживания - ключи, которые будут передаваться в update
            figsize (tuple)
        """
        self.save_dir = save_dir
        self.model_name = model_name
        self.metrics = metrics if metrics is not None else ['loss', 'recall@10']
        self.figsize = figsize
        
        # Хранилище данных: {metric_name: {'epoch': [], 'value': []}}
        self.data = defaultdict(lambda: {'epoch': [], 'value': []})
        
        os.makedirs(save_dir, exist_ok=True)
    
    def update(self, epoch, **kwargs):
        """
        Добавить значения метрик для данной эпохи.
        пример - plotter.update(epoch=5, loss=2.34, recall=0.123, val_loss=2.50, val_recall=0.110)
        """
        for key, value in kwargs.items():
            if value is not None:
                self.data[key]['epoch'].append(epoch)
                self.data[key]['value'].append(value)
    
    def plot(self, show=False, save=True, suffix=''):
        """
        Построить и сохранить график(и).
        
        Args:
            show (bool): показать интерактивное окно
            save (bool): сохранить в файл
            suffix (str): дополнительный суффикс к имени файла
        """
        if not self.data:
            print("Нет данных для построения графиков.")
            return
        
        # Определим, сколько графиков нужно (можно сгруппировать по типу: loss отдельно, метрики отдельно)
        # Простой вариант: все кривые на одном графике (с двумя осями Y, если нужно)
        fig, ax1 = plt.subplots(figsize=self.figsize)
        
        # Разделим метрики на две группы: с префиксом 'val_' и без
        train_metrics = {}
        val_metrics = {}
        for key in self.data.keys():
            if key.startswith('val_'):
                val_metrics[key[4:]] = self.data[key]
            else:
                train_metrics[key] = self.data[key]
        
        # Настроим цвета
        colors = plt.cm.tab10.colors
        color_idx = 0
        
        # Ось Y1 (левая) – для loss
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        
        # Рисуем train loss
        if 'loss' in train_metrics:
            epochs = train_metrics['loss']['epoch']
            values = train_metrics['loss']['value']
            ax1.plot(epochs, values, color='tab:red', linestyle='-', marker='o', label='Train Loss')
        
        # Рисуем val loss
        if 'loss' in val_metrics:
            epochs = val_metrics['loss']['epoch']
            values = val_metrics['loss']['value']
            ax1.plot(epochs, values, color='tab:red', linestyle='--', marker='s', label='Val Loss')
        
        # Ось Y2 (правая) – для метрик качества (Recall, NDCG и т.п.)
        # Создаём вторую ось, если есть хотя бы одна метрика кроме loss
        has_quality_metric = any(k != 'loss' for k in train_metrics) or any(k != 'loss' for k in val_metrics)
        if has_quality_metric:
            ax2 = ax1.twinx()
            ax2.set_ylabel('Metric', color='tab:blue')
            ax2.tick_params(axis='y', labelcolor='tab:blue')
        else:
            ax2 = None
        
        # Функция для рисования метрик качества
        def plot_metric(metric_name, data_dict, is_train=True):
            nonlocal color_idx
            if metric_name in data_dict:
                epochs = data_dict[metric_name]['epoch']
                values = data_dict[metric_name]['value']
                linestyle = '-' if is_train else '--'
                marker = 'o' if is_train else 's'
                label = f"{'Train' if is_train else 'Val'} {metric_name.upper()}"
                color = colors[color_idx % len(colors)]
                color_idx += 1
                if ax2 is not None:
                    ax2.plot(epochs, values, color=color, linestyle=linestyle, marker=marker, label=label)
                else:
                    ax1.plot(epochs, values, color=color, linestyle=linestyle, marker=marker, label=label)
        
        # Рисуем train метрики (кроме loss)
        for metric in train_metrics:
            if metric != 'loss':
                plot_metric(metric, train_metrics, is_train=True)
        
        # Рисуем val метрики (кроме loss)
        for metric in val_metrics:
            if metric != 'loss':
                plot_metric(metric, val_metrics, is_train=False)
        
        # Легенда
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = (ax2.get_legend_handles_labels() if ax2 else ([], []))
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.title(f'Training Curves - {self.model_name}')
        fig.tight_layout()
        
        if save:
            filename = f"{self.model_name}_training_curves{suffix}.png"
            filepath = os.path.join(self.save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"График сохранён: {filepath}")
        
        if show:
            plt.show()
        else:
            plt.close(fig)