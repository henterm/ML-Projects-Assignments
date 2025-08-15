import torch
import argparse
import numpy as np
import json
import os
from data_utils_test import load_test_data, TestDataset
from torch.utils.data import DataLoader
from visualization_n_metrics import calculate_nme, plot_ced
from evaluate import predict_with_model
from collections import defaultdict

def main(data_dir, model_path='best_model.pth'):
    # 1. Инициализация
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. Загрузка модели
    model = torch.load(model_path, map_location=device)
    model.eval()
    
    # 3. Загрузка данных
    test_samples = load_test_data(data_dir)
    test_dataset = TestDataset(test_samples)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # 4. Оценка модели
    errors = defaultdict(list)
    
    for batch in test_loader:
        sample = {
            'image': batch['image'].squeeze(0).to(device),
            'landmarks': batch['landmarks'].squeeze(0),
            'bbox': batch['bbox'].squeeze(0),
            'orig_img': batch['orig_img'][0]
        }
        
        pred = predict_with_model(model, sample, device)
        nme = calculate_nme(pred, sample['landmarks'].cpu().numpy(), 
                           sample['bbox'].cpu().numpy())
        errors['Our Model'].append(nme)
    
    # 5. Визуализация и сохранение результатов
    plot_ced(errors, max_error=0.08, title="Model Evaluation Results")
    
    results = {
        'mean_nme': float(np.mean(errors['Our Model'])),
        'median_nme': float(np.median(errors['Our Model'])),
        'min_nme': float(np.min(errors['Our Model'])),
        'max_nme': float(np.max(errors['Our Model'])),
        'samples': len(errors['Our Model'])
    }
    
    print("\nEvaluation Results:")
    print(json.dumps(results, indent=2))
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate face landmarks detection model')
    parser.add_argument('--data_dir', required=True,
                       help='Path to directory with test images and .pts files')
    parser.add_argument('--model', default='best_model.pth',
                       help='Path to model weights file')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Test directory not found: {args.data_dir}")
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")
    
    main(args.data_dir, args.model)