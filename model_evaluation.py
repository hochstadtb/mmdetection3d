#!/usr/bin/env python
# coding: utf-8

# In[10]:


# Imports
from mmdet3d.apis import init_model, inference_detector
import time

# To convert to a .py file: jupyter nbconvert --to python model_evaluation.ipynb


# In[11]:


# Configuration fields
DISPLAY_VISUALIZATION=True
DEVICE='cpu' # My machine doesn't have an NVIDIA graphics card, so set device to cpu so that we don't attemp to use CUDA cores
CONFIG_FILE='configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
CHECKPOINT='work_dirs/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class/epoch_10.pth'
INPUT_IMAGES=['data/cmpe249-fa22/kitti/testing/velodyne/000000.bin',
              'data/cmpe249-fa22/kitti/testing/velodyne/000001.bin',
              'data/cmpe249-fa22/kitti/testing/velodyne/000002.bin',
              'data/cmpe249-fa22/kitti/testing/velodyne/000003.bin',
              'data/cmpe249-fa22/kitti/testing/velodyne/000004.bin',
              'data/cmpe249-fa22/kitti/testing/velodyne/000005.bin',
              'data/cmpe249-fa22/kitti/testing/velodyne/000006.bin',
              'data/cmpe249-fa22/kitti/testing/velodyne/000007.bin',
              'data/cmpe249-fa22/kitti/testing/velodyne/000008.bin',
              'data/cmpe249-fa22/kitti/testing/velodyne/000009.bin']


# In[12]:


# Initialize the model using the given config file and checkpoint pkl
print('------------------------------')
print('Initializing model using the following..')
print(f'Config file:     {CONFIG_FILE}')
print(f'Checkpoint file: {CHECKPOINT}')
print(f'Device:          {DEVICE}')
print('------------------------------')
model = init_model(CONFIG_FILE, CHECKPOINT, DEVICE)

num_images = len(INPUT_IMAGES)
print(f'Number of inference images on: {num_images}')

time_deltas = []
for image in INPUT_IMAGES:
    print(f'Performing inference on {image}..')

    start_time = time.time()
    results, data = inference_detector(model, image)
    time_delta = time.time() - start_time
    
    print(f'Completed, latency of {time_delta}s')
    time_deltas.append(time_delta)

    model.show_results(data, results, out_dir='results', show=DISPLAY_VISUALIZATION)

inference_time_total = sum(time_deltas)
print('------------------------------')
print(f'Total inference latency:   {inference_time_total}s')
print(f'Average inference latency: {inference_time_total / num_images}s')

