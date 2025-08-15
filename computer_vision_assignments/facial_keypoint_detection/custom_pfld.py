import torch
import torch.nn as nn
from pfld import PFLDInference

class CustomPFLD(nn.Module):
    def __init__(self):
        super().__init__()
        # Основная сеть PFLD
        self.backbone = PFLDInference()

        # Дополнительные слои для получения 136 выходов (68 точек × 2 координаты)
        self.adjust_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 136)
        )

    def forward(self, x):
        # Получаем features от PFLD
        features, _ = self.backbone(x)

        # Преобразуем к нужному размеру
        outputs = self.adjust_layer(features)
        return outputs