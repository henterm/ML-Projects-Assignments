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

# Загрузка из репозитория модели PFLD
"""
if not os.path.exists('PFLD-pytorch'):
    !git clone https://github.com/polarisZhao/PFLD-pytorch.git
    %cd PFLD-pytorch
else:
    %cd PFLD-pytorch

import sys
sys.path.append('/content/PFLD-pytorch')
from models.pfld import PFLDInference
%cd /content
"""

# Инициализация детектора и предиктора DLIB
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Функция загрузки точек
def load_pts(pts_path):
    try:
        with open(pts_path) as f:
            lines = [line.strip() for line in f.readlines()]

        points = []
        start = False
        for line in lines:
            if line == "{":
                start = True
                continue
            if line == "}":
                break
            if start and line:
                x, y = map(float, line.split())
                points.append((x, y))

        return np.array(points) if len(points) in [68, 39] else None
    except:
        return None

# Обработка одного образца
def process_sample(img_path, pts_path):
    try:
        img = cv2.imread(img_path)
        if img is None:
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        landmarks = load_pts(pts_path)

        if landmarks is None or len(landmarks) not in [68, 39]:
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

        # Нормализация landmarks
        norm_landmarks = []
        for (x, y) in landmarks:
            nx = np.clip((x - x1) / (x2 - x1), 0.0, 1.0)
            ny = np.clip((y - y1) / (y2 - y1), 0.0, 1.0)
            norm_landmarks.append([nx, ny])

        norm_landmarks = np.array(norm_landmarks, dtype=np.float32)

        return {
            'image': face_resized,
            'landmarks': norm_landmarks,
            'bbox': [x1, y1, x2, y2],
            'orig_size': (w, h)
        }
    except Exception as e:
        print(f"Error in {img_path}: {str(e)}")
        return None


def load_processed_data(processed_dir):
    samples = []
    for data_path in glob(os.path.join(processed_dir, '*.npy')):
        img_path = data_path.replace('.npy', '.jpg')
        if os.path.exists(img_path):
            try:
                data = np.load(data_path, allow_pickle=True).item()
                if 'image' in data:  # Проверяем новый формат
                    samples.append((img_path, data_path))
            except:
                continue
    return samples


# Подготовка данных
def prepare_data(dataset_paths, force_reprocess=False):
    processed_dir = 'processed'
    os.makedirs(processed_dir, exist_ok=True)

    # Проверяем, есть ли уже обработанные данные в новом формате
    existing_samples = []
    if not force_reprocess:
        existing_samples = load_processed_data(processed_dir)
        # Дополнительная проверка формата данных
        if existing_samples:
            sample_data = np.load(existing_samples[0][1], allow_pickle=True).item()
            if 'image' not in sample_data:  # Проверяем новый формат
                print("Found old format data, reprocessing...")
                existing_samples = []

    if existing_samples:
        print(f"Found {len(existing_samples)} preprocessed samples in new format. Loading...")
        return existing_samples

    print("Preprocessing data in new format...")
    samples = []

    for dataset_path in dataset_paths:
        img_paths = glob(f"{dataset_path}/**/*.jpg", recursive=True)
        for img_path in tqdm(img_paths, desc=f"Processing {dataset_path}"):
            pts_path = os.path.splitext(img_path)[0] + '.pts'
            if not os.path.exists(pts_path):
                continue

            result = process_sample(img_path, pts_path)
            if result:
                base_name = os.path.basename(img_path)
                save_data_path = f"{processed_dir}/{base_name.replace('.jpg', '.npy')}"

                np.save(save_data_path, result)

                cv2.imwrite(
                    f"{processed_dir}/{base_name}",
                    cv2.cvtColor(result['image'], cv2.COLOR_RGB2BGR)
                )

                samples.append((f"{processed_dir}/{base_name}", save_data_path))

    print(f"\nSuccessfully processed {len(samples)} samples in new format")

    # Сразу сохраняем в архив
    if samples:
      print("Creating backup archive...")
      try:
          # 1. Сначала создаем архив локально
          shutil.make_archive('processed', 'zip', processed_dir)

          # 2. Проверяем, что архив создан
          if not os.path.exists('processed.zip'):
              raise RuntimeError("Archive creation failed!")

          print(f"Local archive created: {os.path.getsize('processed.zip')/1024/1024:.2f} MB")

          # 3. Копируем в Google Drive (если подключен)
          if os.path.exists('/content/drive'):
              drive_target_dir = '/content/drive/MyDrive/Facial_Keypoint'
              os.makedirs(drive_target_dir, exist_ok=True)
              shutil.copy('processed.zip', drive_target_dir)
              print(f"Copied to Google Drive: {drive_target_dir}")
          else:
              print("Google Drive not mounted, archive saved locally only")

      except Exception as e:
          print(f"Error during archiving: {str(e)}")
          # Можно попробовать сохранить локально
          if os.path.exists(processed_dir):
              shutil.make_archive('processed_fallback', 'zip', processed_dir)
              print("Created fallback archive: processed_fallback.zip")

    return samples


# Датасет
class FaceLandmarksDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples  # Сохраняем исходные пары (img_path, data_path)
        self.transform = transform
        self.valid_indices = []  # Индексы валидных образцов
        self.error_count = 0

        # Предварительная проверка данных
        for idx in tqdm(range(len(samples)), desc="Validating dataset"):
            try:
                data = np.load(samples[idx][1], allow_pickle=True).item()
                landmarks = data['landmarks']

                # Проверяем размер landmarks
                if landmarks.shape != (68, 2):
                    self.error_count += 1
                    continue

                self.valid_indices.append(idx)
            except Exception as e:
                self.error_count += 1
                continue

        print(f"\nSuccessfully loaded {len(self.valid_indices)} samples (skipped {self.error_count} invalid)")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        img_path, data_path = self.samples[real_idx]

        data = np.load(data_path, allow_pickle=True).item()
        image = data['image'].astype(np.float32) / 255.0
        landmarks = data['landmarks'].astype(np.float32).flatten()

        if self.transform:
            transformed = self.transform(image=image, keypoints=landmarks.reshape(-1, 2))
            image = transformed['image']
            landmarks = transformed['keypoints'].flatten()

        return {
            'image': torch.tensor(image).permute(2, 0, 1),
            'landmarks': torch.tensor(landmarks),
            'bbox': data.get('bbox', [0, 0, image.shape[1], image.shape[0]]),
            'orig_size': data.get('orig_size', (image.shape[1], image.shape[0]))
        }