import numpy as np 
ratios=[0.5,1,2]
anchor_scales=[8,16,32]
sub_sample=16
fe_size = (800//16)
ctr_x = np.arange(16, (fe_size+1) * 16, 16)
ctr_y = np.arange(16, (fe_size+1) * 16, 16)
num_combinations = len(ctr_x) * len(ctr_y)
ctr = np.zeros((num_combinations, 2))
index = 0
for x in range(len(ctr_x)):
    for y in range(len(ctr_y)):
        ctr[index, 1] = ctr_x[x] - 8
        ctr[index, 0] = ctr_y[y] - 8
        index +=1
anchors = np.zeros(((fe_size * fe_size)*9, 4))
index = 0
for c in ctr:
  ctr_y, ctr_x = c
  for i in range(len(ratios)):
    for j in range(len(anchor_scales)):
      h = sub_sample * anchor_scales[j] * np.sqrt(ratios[i])
      w = sub_sample * anchor_scales[j] * np.sqrt(1./ ratios[i])
      anchors[index, 0] = ctr_y - h / 2.
      anchors[index, 1] = ctr_x - w / 2.
      anchors[index, 2] = ctr_y + h / 2.
      anchors[index, 3] = ctr_x + w / 2.
      index += 1
anchors.shape