import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import auc


# Функция для расчета ошибки
def calculate_nme(pred_landmarks, true_landmarks, bbox):
    """
    Calculate Normalized Mean Error (NME)
    :param pred_landmarks: predicted landmarks (68, 2)
    :param true_landmarks: ground truth landmarks (68, 2)
    :param bbox: bounding box [x1, y1, x2, y2]
    :return: NME
    """
    # Размер bbox для нормализации
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    norm_factor = np.sqrt(w * h)

    # Среднеквадратичная ошибка
    mse = np.mean(np.sqrt(np.sum((pred_landmarks - true_landmarks) ** 2, axis=1)))

    # Нормализованная ошибка
    nme = mse / norm_factor
    return nme


# Функция для построения CED графиков
def plot_ced(errors_dict, max_error=0.08, title="Cumulative Error Distribution"):
    plt.figure(figsize=(10, 6))

    for method, errors in errors_dict.items():
        # Сортировка ошибок
        sorted_errors = np.sort(errors)

        # Рассчет CED
        y = np.arange(1, len(sorted_errors)+1) / len(sorted_errors)
        x = sorted_errors

        # Обрезаем до max_error
        mask = x <= max_error
        x = x[mask]
        y = y[mask]

        # Рассчет AUC
        auc_score = auc(x, y) / max_error

        plt.plot(x, y, label=f"{method} (AUC: {auc_score:.3f})")

    plt.xlabel("Normalized Mean Error")
    plt.ylabel("Fraction of Images")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.xlim(0, max_error)
    plt.ylim(0, 1)
    plt.show()


class WingLoss(nn.Module):
    def __init__(self, w=10.0, epsilon=2.0):
        super().__init__()
        self.w = torch.tensor(w, dtype=torch.float32)
        self.epsilon = torch.tensor(epsilon, dtype=torch.float32)

    def forward(self, pred, target):
        device = pred.device
        self.w = self.w.to(device)
        self.epsilon = self.epsilon.to(device)

        x = (pred - target).abs()
        c = self.w - self.w * torch.log(1.0 + self.w/self.epsilon)
        loss = torch.where(
            x < self.w,
            self.w * torch.log(1.0 + x/self.epsilon),
            x - c
        )
        return loss.mean()