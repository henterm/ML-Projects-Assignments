# Face Landmarks Detection

## Installation
1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Download dlib model: `shape_predictor_68_face_landmarks.dat`

## Usage
- Training: `python train.py --data_dir path/to/train_data`
- Evaluation: `python evaluate.py --test_dir path/to/test_data`
- Run demo: `python RunModel.py --data_dir path/to/test_data`

## Model Weights
Place pretrained model `best_model.pth` in the root directory