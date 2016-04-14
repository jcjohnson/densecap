#DenseCap

This is the code for the paper

*[DenseCap: Fully Convolutional Localization Networks for Dense Captioning](http://cs.stanford.edu/people/jcjohns/papers/densecap/JohnsonCVPR2016.pdf)*, [Justin Johnson](http://cs.stanford.edu/people/jcjohns/)\*, [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/)\*, [Li Fei-Fei](http://vision.stanford.edu/feifeili/), CVPR 2016 (\* indicates equal contribution)

The paper addresses the problem of **dense captioning**, where detect objects in images and describe them in natural language. Here are a few example outputs:

Dependencies: 

- [stnbhwd](https://github.com/qassemoquab/stnbhwd) so that you can `require 'stn'`


To train, example:

```
$ th main.lua -data_h5 [location] -data_json [location]
```
