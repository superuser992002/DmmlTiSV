Deep multi-metric learning for text-independent speaker verification
====  

By Jiwei Xu, Xinggang Wang, Bin Feng, Wenyu Liu.

This code is a implementation of the experiments on Voxceleb 1 and Voxceleb 2

Our method achieved an EER of 3.48. [**model-3.48 link**](https://share.weiyun.com/5yCYqCC)

We randomly add some noise signals to the training data during the training process as our data enhancement method. [**noise link**](http://www.openslr.org/28/)




Dependencies
====  

Python 3.6

Pytorch 1.2

librosa

scipy

soundfile

python_speech_features


Download Dataset
====  
[Voxceleb 1/2](http://blog.csdn.net/guodongxiaren) corpus can be downloaded directly from the official website.


Preprocess data
------- 

First convert the .m4a file to a .wav file

```
cd convert_data
sh convert.sh
```


Train model
------- 

```
python train.py
```

Test model
------- 

```
python test.py
```

Thanks to the Third Party Libs
====
[metric_learning](https://github.com/tomp11/metric_learning)
