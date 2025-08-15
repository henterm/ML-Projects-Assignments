import torch
import dlib
import numpy as np
import cv2
import glob
import os
import shutil
from tqdm import tqdm
from pfld import PFLDInference
from torch.utils.data import Dataset, DataLoader
from data_utils_train import load_pts
from pathlib import Path


# Инициализация детектора и предиктора DLIB
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Функция для загрузки тестовых данных
def load_test_data(test_dir):
    samples = []
    pts_files = list(Path(test_dir).rglob("*.pts"))

    for pts_path in pts_files:
        with open(pts_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if 'n_points' in line:
                    n_points = int(line.strip().split(':')[1])
                    break

            # Пропускаем изображения с 39 точками
            if n_points == 39:
                continue

            img_path = pts_path.with_suffix('.jpg')
            if img_path.exists():
                samples.append((str(img_path), str(pts_path)))

    return samples

# Функция для обработки тестового образца (аналогично train, но без нормализации)
def process_test_sample(img_path, pts_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        landmarks = load_pts(pts_path)

        if landmarks is None or len(landmarks) != 68:
            return None

        # Детекция лица
        dets = detector(img, 1)
        if not dets:
            return None

        bbox = dets[0]
        x1, y1, x2, y2 = bbox.left(), bbox.top(), bbox.right(), bbox.bottom()
        w, h = x2 - x1, y2 - y1

        # Увеличиваем margin (30%)
        margin = 0.3
        x1 = max(0, int(x1 - margin * w))
        y1 = max(0, int(y1 - margin * h))
        x2 = min(img.shape[1], int(x2 + margin * w))
        y2 = min(img.shape[0], int(y2 + margin * h))

        if x1 >= x2 or y1 >= y2:
            return None

        # Crop и resize до 112x112
        face_crop = img[y1:y2, x1:x2]
        face_resized = cv2.resize(face_crop, (112, 112))

        return {
            'image': face_resized,
            'landmarks': landmarks,  # Оригинальные координаты (не нормализованные)
            'bbox': [x1, y1, x2, y2],
            'orig_img': img,
            'img_path': img_path
        }
    except Exception as e:
        print(f"Error in {img_path}: {str(e)}")
        return None

# Датасет для тестирования
class TestDataset(Dataset):
    def __init__(self, samples):
        self.samples = []
        self.invalid_count = 0

        for img_path, pts_path in tqdm(samples, desc="Loading test data"):
            data = process_test_sample(img_path, pts_path)
            if data is not None:
                self.samples.append(data)
            else:
                self.invalid_count += 1

        print(f"Loaded {len(self.samples)} valid samples, skipped {self.invalid_count} invalid")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = self.samples[idx]

        # Нормализация изображения для модели
        image = data['image'].astype(np.float32) / 255.0
        image = torch.tensor(image).permute(2, 0, 1)  # (C, H, W)

        return {
            'image': image,
            'landmarks': torch.tensor(data['landmarks']),
            'bbox': torch.tensor(data['bbox']),
            'orig_img': data['orig_img'],
            'img_path': data['img_path']
        }