from __future__ import print_function
import argparse
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
import glob
import random
from torchvision import models
from PIL import Image
from torch.optim import lr_scheduler
import time
from data_loader import transforms, folder_noise, dataloader, Sampler

#from data_loader import 
from models import resnet

#from losses.TripletLoss import TripletLoss
from losses.losses import TripletLoss, NpairLoss, AngularLoss
from scipy.stats import norm
from config import Config
from torch.autograd import Variable

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

opt = Config()

model = resnet.resnet50(num_features=2048, dropout=0.1, num_classes=opt.num_classes).cuda()
model.load_state_dict(torch.load("/content/drive/My Drive/multi-metric/model_latest.pth"))
model = nn.DataParallel(model)



def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(- target * F.log_softmax(logits, -1), -1))
    else:
        return torch.sum(torch.sum(- target * F.log_softmax(logits, -1), -1))




def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)

    normp = torch.sum(buffer, 1).add_(1e-10)
    norm = torch.sqrt(normp)

    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output


class Dataset(object):

    def __init__(self, x0, x1, x00):
        self.size = x0.shape[0]
        self.x0 = x0
        self.x1 = x1

    def __getitem__(self, index):
        return (self.x0[index],
                self.x1[index])

    def __len__(self):
        return self.size



data_dir = opt.data_dir
im_datasets = NpyFolder(data_dir)
print('load')
train_loader  = DataLoader(im_datasets, batch_size=opt.ims_ids*opt.ims_per_id, sampler=Sampler.PKSampler(im_datasets),shuffle=False,num_workers=5)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred    = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


#torch.cuda.set_device(1)
def train_model(n_epochs=25):
    since = time.time()
    #best_model_wts = model.state_dict()
    best_acc = 0.0
    best_loss = float('inf')


    for epoch in range(n_epochs):
        print (" | Epoch {}/{}".format(epoch, n_epochs-1))
        print (" | " + "-" * 20)


        model.train(True)

        running_loss = 0.0
        running_corrects = 0

        for step, datas in enumerate(train_loader): 

            inputs, labels = datas

            inputs = inputs.float()
            inputs = Variable(inputs.cuda())       
            labels = Variable(labels.cuda())

            logits, features = model(inputs)

            anchor = features[::opt.ims_per_id]
            positive = features[1::opt.ims_per_id]
            target = labels[::opt.ims_per_id]

            npair_loss = criterion_pair(anchor, positive, target)


            #npair_loss = criterion_pair(features, labels)
            angular_loss = criterion_angular(features, labels)


            prec1, prec5 = accuracy(logits, labels, topk=(1, 5))

            loss_softmax = criterion(logits, labels)
            
            loss_tri, prec_tri = criterion_tri(features, labels)

            loss = 0.1*loss_softmax + loss_tri + angular_loss + 0.5*npair_loss


            optimizer.zero_grad()
 

            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=2, norm_type=2)
            #optimizer1.step()
            optimizer.step()

            loss_softmax = loss_softmax.data.cpu() 

            if step % 1 == 0:
                print('Train Epoch: {} [{}/{}]\t'
                    'acc: {:.4f} \t'
                    'tri_loss: {:.4f} \t'
                    'npair_loss: {:.4f} \t'
                    'angular_loss: {:.4f} \t'
                    'softmax_loss: {:.4f} \t'.format(
                epoch, step, len(train_loader),
                prec1, loss_tri, npair_loss, angular_loss, loss_softmax))

            if step % 40 == 0:
   
                torch.save(model.state_dict(),"/content/drive/My Drive/multi-metric/model_latest.pth")
                print (" | Time consuming: {:.2f}s".format(time.time()-since))
                print (" | ")


        torch.save(model.state_dict(),"/content/drive/My Drive/multi-metric/model_{}.pth".format(epoch))

    print (" | Time consuming: {:.2f}s".format(time.time()-since))
    print (" | ")
        

if __name__ == '__main__':


    model = model.cuda()
    criterion_tri = TripletLoss(margin=0.5, num_instances=opt.ims_per_id).cuda()
    criterion_pair = NpairLoss()  
    criterion_angular = AngularLoss()
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(),lr=0.00003, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    model = train_model(n_epochs=1000)





