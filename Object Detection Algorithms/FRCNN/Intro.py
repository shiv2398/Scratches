import torch 
from plotter import plotter
image=torch.zeros((1,3,800,800)).float()
bbox = torch.FloatTensor([[20, 30, 400, 500], [200, 300, 400, 400]]) # [y1, x1, y2, x2] format
labels = torch.LongTensor([6, 8]) # 0 represents background
plotter(image,bbox)