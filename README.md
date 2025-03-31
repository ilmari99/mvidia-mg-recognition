# Micro Gesture Classification with iMiGUE Dataset


## The project structure
- `analysis.ipynb`: A notebook that contains some preliminary analysis of the dataset.
- `inference.py`: A Python script that contains code for inference. Example usage is such as
```
python inference.py --data_path data/test --checkpoint_path Swin3D.ckpt
```
- `main.py`: A python script that contains the code for training image classification, late fusion, and video classification models.
- `engine.py`: A Python file Pytorch Lighting modules for the models 
- `utils/dataset.py`: A Python file that contains custom image and video datasets for iMiGUE 
- `utils/misc.py`: A Python file that contains varying functions.

## Model weights
- One can download the weights for Swin3D from this [link](https://lut-my.sharepoint.com/:u:/g/personal/joona_kareinen_lut_fi/EQuW3dXT2YJHlxaLhDO-CGYBLS07wHIRfMmiVU4e6KbHUg?e=3lMzft).