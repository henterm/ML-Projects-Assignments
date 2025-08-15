import torch
import torch.nn as nn
import torch.optim as optim
import dlib
import numpy as np
import cv2
import glob
import os
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pfld import PFLDInference
from torch.utils.data import Dataset, DataLoader
from data_utils_train import FaceLandmarksDataset
from custom_pfld import CustomPFLD
from visualization_n_metrics import WingLoss

def train_model(samples, num_workers=2):

    train_samples, val_samples = train_test_split(samples, test_size=0.2, random_state=42)

    # Создание датасетов
    train_dataset = FaceLandmarksDataset(train_samples)
    val_dataset = FaceLandmarksDataset(val_samples)

    # Функция для объединения образцов в батчи
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        landmarks = torch.stack([item['landmarks'] for item in batch])
        bboxes = [item['bbox'] for item in batch]
        orig_sizes = [item['orig_size'] for item in batch]

        return {
            'images': images,
            'landmarks': landmarks,
            'bboxes': bboxes,
            'orig_sizes': orig_sizes
        }

    # Создание DataLoader'ов
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True
    )

    # Инициализация модели и оптимизатора
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomPFLD().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = WingLoss(w=5, epsilon=1.0).to(device)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    best_loss = float('inf')
    early_stopping_counter = 0
    patience = 100

    # Проверка данных
    def check_data_quality(dataset, name):
        error_count = 0
        print(f"\nПроверка качества данных ({name}):")

        sample = dataset[0]
        print("Первый образец:")
        print("Тип:", type(sample))
        print("Ключи:", sample.keys())
        print("Изображение:", sample['image'].shape, sample['image'].dtype)
        print("Landmarks:", sample['landmarks'].shape, sample['landmarks'].dtype)

        for i in range(min(100, len(dataset))):
            try:
                sample = dataset[i]
                if sample['image'].sum() == 0 or sample['landmarks'].sum() == 0:
                    print(f"Sample {i} содержит нулевые данные!")
                    error_count += 1
            except Exception as e:
                print(f"Ошибка в sample {i}: {str(e)}")
                error_count += 1

        print(f"Найдено {error_count} проблемных образцов из {min(100, len(dataset))} проверенных")

    check_data_quality(train_dataset, "train")
    check_data_quality(val_dataset, "val")

    # Проверка батча
    print("\nПроверка батча из train_loader:")
    batch = next(iter(train_loader))
    print("Изображения:", batch['images'].shape, batch['images'].dtype)
    print("Landmarks:", batch['landmarks'].shape, batch['landmarks'].dtype)

    # Визуализация образца
    sample = train_dataset[7]
    img = sample['image'].permute(1, 2, 0).numpy()
    landmarks = sample['landmarks'].reshape(-1, 2) * 112
    plt.imshow(img)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c='r', s=5)
    plt.show()

    # Цикл обучения
    for epoch in range(100):
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = batch['images'].to(device)
            targets = batch['landmarks'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        # Валидация
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images = batch['images'].to(device)
                targets = batch['landmarks'].to(device)

                outputs = model(images)
                val_loss += criterion(outputs, targets).item()

                if epoch % 5 == 0:  # Визуализиция первого образца каждые 5 эпох
                    img = images[0].cpu().permute(1, 2, 0).numpy()
                    pred = outputs[0].view(-1, 2).cpu().numpy()
                    true = targets[0].view(-1, 2).cpu().numpy()

                    # Денормализация
                    h, w = img.shape[:2]
                    pred = pred * [w, h]
                    true = true * [w, h]

                    plt.figure(figsize=(10, 5))
                    plt.subplot(121)
                    plt.imshow(img)
                    plt.scatter(pred[:, 0], pred[:, 1], c='red', s=5, label='Pred')
                    plt.title("Predicted")

                    plt.subplot(122)
                    plt.imshow(img)
                    plt.scatter(true[:, 0], true[:, 1], c='green', s=5, label='True')
                    plt.title("Ground Truth")
                    plt.show()
                    break

        # Расчет потерь
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Сохранение модели
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Ранняя остановка на эпохе {epoch+1}")
                break

    print("Обучение завершено!")
    return model