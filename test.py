import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import io
import requests

from PIL import Image
from torchvision import models
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
#import cv2
import pdb
import torch
import torch.nn as nn
import glob
import random
import numpy as np
import time
import math
import sys
import scipy.io as sio
from sklearn import *
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from torch.utils import data
from tqdm import tqdm
from config import Config
from sklearn import *
from sklearn import metrics as Metrics
from models import resnet
from torch.nn import DataParallel
import pandas as pd
import librosa
from scipy.signal.windows import hamming
import soundfile as sf
from numpy import dot
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity

def npy_loader(path):
    npy = np.load(path)
    return npy

def cosin_metric(a, b):
    cos_sim = cosine_similarity(a,b)
    return cos_sim

def TEST_ZALO(model):
  test = pd.read_csv("/content/self_building_test_zalo.csv") 
  labels = test.label.values
  ut_1s = test.left.values
  ut_2s = test.right.values
  for idx, (ut_1, ut_2) in enumerate(zip(ut_1s, ut_2s)):
    ut_1 = npy_loader(ut_1)
    ut_2 = npy_loader(ut_2)
    feature_1 = model(torch.unsqueeze(torch.FloatTensor(ut_1), 0).cuda())
    feature_2 = model(torch.unsqueeze(torch.FloatTensor(ut_2), 0).cuda())
    feature_1 = feature_1.detach().cpu().numpy().reshape(1,1024)
    feature_2 = feature_2.detach().cpu().numpy().reshape(1,1024)
    score = cosin_metric(feature_1, feature_2)


    if score >= 0.5:
        label = 1
    else:
        label = 0
    test.at[idx, "score"] = float(score)
    test.at[idx, "label_predict"] = label
  return accuracy_score(labels, test.label_predict), test.score