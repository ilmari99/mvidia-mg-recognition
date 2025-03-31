# Micro Gesture Classification with iMiGUE Dataset

## The project structure
- `analysis.ipynb`: A notebook that contains preliminary analysis of the dataset.
- `inference.py`: A Python script that contains code for inference.
- `main.py`: A python script that contains the code for training image classification, late fusion, and video classification models.
- `engine.py`: A Python file Pytorch Lighting modules for the models 
- `utils/dataset.py`: A Python file that contains custom image and video datasets for iMiGUE 
- `utils/misc.py`: A Python file that contains varying functions.

## To test our model
### 1. Clone the repository
```
git clone https://github.com/ilmari99/mvidia-mg-recognition
cd mvidia-mg-recognition
```
### 2. Download the weights
Download the weights for our finetuned Swin3D from https://lut-my.sharepoint.com/:u:/g/personal/joona_kareinen_lut_fi/EQuW3dXT2YJHlxaLhDO-CGYBLS07wHIRfMmiVU4e6KbHUg?e=3lMzft

Move the file to the root of the repository.
### 3. Install the required packages:
```
pip install -r requirements.txt
```
### 4. Run inference on the test set:
```
python inference.py --data_path data/test --checkpoint_path Swin3D.ckpt
```
This will produce a report such as (these results are on the train set):
```
───────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
───────────────────────────────────────────────────────────────────────────
        test_acc            0.7471148371696472
        test_loss           0.8477447628974915
      test_top5_acc         0.9738207459449768
───────────────────────────────────────────────────────────────────────────
```