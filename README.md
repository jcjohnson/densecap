#DenseCap

This is the code for the paper

*[DenseCap: Fully Convolutional Localization Networks for Dense Captioning](http://cs.stanford.edu/people/jcjohns/papers/densecap/JohnsonCVPR2016.pdf)*, [Justin Johnson](http://cs.stanford.edu/people/jcjohns/)\*, [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/)\*, [Li Fei-Fei](http://vision.stanford.edu/feifeili/), [CVPR 2016](http://cvpr2016.thecvf.com/) (\* indicates equal contribution)

The paper addresses the problem of **dense captioning**, where a computer detects objects in images and describes them in natural language. Here are a few example outputs:

<img src='imgs/resultsfig.png'>

The model is a deep convolutional neural network trained in an end-to-end fashion on the [Visual Genome](https://visualgenome.org/) dataset.

With this repository you can:

- Run our pretrained DenseCap model on new images, on GPU or CPU
- Train a new DenseCap model on your own data
- Run a live demo with a trained DenseCap model using a webcam

## Installation

You can find full [installation instructions here](doc/INSTALL.md).

## Pretrained model

You can download a pretrained DenseCap model by running the following script:

```bash
 sh scripts/download_pretrained_model.sh
 ```
 
 This will download a zipped version of the model (about 1.1 GB) to `data/models/densecap/densecap-pretrained-vgg16.t7.zip`, unpack
 it to `data/models/densecap/densecap-pretrained-vgg16.t7` (about 1.2 GB) and then delete the zipped version.
 
 This is not the exact model that was used in the paper, but is has comparable performance; using 1000 region proposals per image,
 it achieves a mAP of 5.70 on the test set which is slightly better than the mAP of 5.39 that we report in the paper.
