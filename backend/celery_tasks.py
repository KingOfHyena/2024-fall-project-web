from celery import Celery
from model import MyCNN
from dataset import MyDataset
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

app = Celery('tasks', broker='redis://localhost:6379/0')

MODEL_PATH = 'model/cnn_10_750_1.0.model'
HEATMAP_FOLDER = './static/heatmap/'
cnn = torch.load(MODEL_PATH)
cnn.eval()

@app.task
def generate_heatmap(data_path, label):
    dataset = MyDataset([data_path])
    sample = dataset.select()
    data = sample['data'].unsqueeze(0)  # 增加batch维度

    output = cnn(data)
    _, pred = torch.max(output, 1)
    
    data_np = data.squeeze().numpy()
    heatmap_path = os.path.join(HEATMAP_FOLDER, f'heatmap_{label}.png')
    plt.imshow(data_np, cmap='viridis', interpolation='spline16')
    plt.colorbar()
    plt.savefig(heatmap_path)
    plt.close()

    return heatmap_path, int(pred)
