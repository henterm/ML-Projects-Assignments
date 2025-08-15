import torch
import numpy as np
import dlib
import matplotlib.pyplot as plt
import torch.nn as nn
import json
from tqdm import tqdm
from collections import defaultdict
from visualization_n_metrics import calculate_nme, plot_ced
from data_utils_test import load_test_data, TestDataset
from torch.utils.data import Dataset, DataLoader

# Инициализация детектора и предиктора DLIB
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Функция для предсказания с помощью модели
def predict_with_model(model, sample, device):
    model.eval()
    with torch.no_grad():
        # Убедимся, что изображение имеет правильную размерность (C, H, W)
        image = sample['image'].to(device)

        # Добавляем batch dimension если его нет
        if image.dim() == 3:
            image = image.unsqueeze(0)  # (1, C, H, W)

        pred = model(image).cpu().numpy().reshape(-1, 2)

        # Денормализация предсказанных точек к оригинальному размеру 112x112
        pred = pred * 112

        # Масштабирование точек к оригинальному bbox
        bbox = sample['bbox'].cpu().numpy()
        w_scale = (bbox[2] - bbox[0]) / 112
        h_scale = (bbox[3] - bbox[1]) / 112

        pred[:, 0] = pred[:, 0] * w_scale + bbox[0]
        pred[:, 1] = pred[:, 1] * h_scale + bbox[1]

        return pred

# Функция для предсказания с помощью DLIB
def predict_with_dlib(sample):
    try:
        # Преобразуем изображение из тензора в numpy array
        if isinstance(sample['orig_img'], torch.Tensor):
            orig_img = sample['orig_img'].cpu().numpy()
        else:
            orig_img = sample['orig_img']

        # Убедимся, что изображение в правильном формате (uint8, BGR)
        if orig_img.dtype != np.uint8:
            orig_img = (orig_img * 255).astype(np.uint8)
        if len(orig_img.shape) == 3 and orig_img.shape[0] == 3:
            orig_img = orig_img.transpose(1, 2, 0)
        if orig_img.shape[2] == 3:  # RGB -> BGR
            orig_img = orig_img[:, :, ::-1]

        # Преобразуем bbox в dlib.rectangle
        bbox = sample['bbox']
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.cpu().numpy()
        dlib_rect = dlib.rectangle(*bbox.astype(int))

        # Предсказание ключевых точек
        shape = predictor(orig_img, dlib_rect)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        return landmarks
    except Exception as e:
        print(f"DLIB prediction failed: {str(e)}")
        return None
    
# Основная функция тестирования
def evaluate_model(model, test_dirs, device):
    # Загрузка тестовых данных
    test_samples = []
    for test_dir in test_dirs:
        test_samples += load_test_data(test_dir)

    test_dataset = TestDataset(test_samples)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )

    # Словари для хранения ошибок
    errors = defaultdict(lambda: defaultdict(list))
    dlib_fail_count = 0

    # Тестирование
    for batch in tqdm(test_loader, desc="Evaluating"):
        img_path = batch['img_path'][0]
        dataset_name = "300W" if "300W" in img_path else "Menpo"

        # Подготовка данных
        sample = {
            'image': batch['image'].squeeze(0).to(device),
            'landmarks': batch['landmarks'].squeeze(0),
            'bbox': batch['bbox'].squeeze(0),
            'orig_img': batch['orig_img'][0]
        }

        # Предсказание нашей моделью
        pred_landmarks_model = predict_with_model(model, sample, device)
        true_landmarks = sample['landmarks'].cpu().numpy()
        bbox = sample['bbox'].cpu().numpy()

        # Расчет ошибки для модели
        nme_model = calculate_nme(pred_landmarks_model, true_landmarks, bbox)
        errors[dataset_name]['Our Model'].append(nme_model)

        # Только для Menpo сравниваем с DLIB
        if dataset_name == "Menpo":
            pred_landmarks_dlib = predict_with_dlib(sample)
            if pred_landmarks_dlib is not None:
                nme_dlib = calculate_nme(pred_landmarks_dlib, true_landmarks, bbox)
                errors[dataset_name]['DLIB'].append(nme_dlib)
            else:
                dlib_fail_count += 1

    print(f"\nDLIB failed to predict on {dlib_fail_count} samples")

    # Построение графиков
    for dataset_name in errors:
        plot_ced(errors[dataset_name],
                max_error=0.08,
                title=f"CED Curve - {dataset_name} Dataset")

    # Сохранение результатов
    results = {}
    for dataset_name in errors:
        results[dataset_name] = {}
        for method in errors[dataset_name]:
            errors_array = np.array(errors[dataset_name][method])
            results[dataset_name][method] = {
                'mean': np.mean(errors_array),
                'std': np.std(errors_array),
                'median': np.median(errors_array),
                'max': np.max(errors_array),
                'min': np.min(errors_array),
                'samples': len(errors_array)
            }

    print("\nResults Summary:")
    print(json.dumps(results, indent=4))

    return results