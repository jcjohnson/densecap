#DenseCap

This is the code for the paper

*[DenseCap: Fully Convolutional Localization Networks for Dense Captioning](http://cs.stanford.edu/people/jcjohns/papers/densecap/JohnsonCVPR2016.pdf)*, [Justin Johnson](http://cs.stanford.edu/people/jcjohns/)\*, [Andrej Karpathy](http://cs.stanford.edu/people/karpathy/)\*, [Li Fei-Fei](http://vision.stanford.edu/feifeili/), [CVPR 2016](http://cvpr2016.thecvf.com/) (\* indicates equal contribution)

The paper addresses the problem of **dense captioning**, where a computer detects objects in images and describes them in natural language. Here are a few example outputs:

<img src='imgs/resultsfig.png'>

The model is trained using the [Visual Genome](https://visualgenome.org/) dataset, and the model is a deep convolutional neural network
that is trained in an end-to-end fashion.

With this repository you can:

- Run our pretrained DenseCap model on new images, on GPU or CPU
- Train a new DenseCap model on your own data
- Run a live demo with a trained DenseCap model using a webcam

## Installation

DenseCap is implemented in [Torch](http://torch.ch/), and depends on the following packages:

- [torch/torch7](https://github.com/torch/torch7)
- [torch/nn](https://github.com/torch/nn)
- [torch/nngraph](https://github.com/torch/nngraph)
- [torch/image](https://github.com/torch/image)
- [lua-cjson](https://luarocks.org/modules/luarocks/lua-cjson)
- [qassemoquab/stnbhwd](https://github.com/qassemoquab/stnbhwd)
- [jcjohnson/torch-rnn](https://github.com/jcjohnson/torch-rnn)

After installing torch, you can install / update these dependencies by running the following:

```bash
luarocks install torch
luarocks install nn
luarocks install image
luarocks install lua-cjson
luarocks install https://raw.githubusercontent.com/qassemoquab/stnbhwd/master/stnbhwd-scm-1.rockspec
luarocks install https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/torch-rnn-scm-1.rockspec
```

### (Optional) GPU acceleration

If have an NVIDIA GPU and want to accelerate the model with CUDA, you'll also need to install
[torch/cutorch](https://github.com/torch/cutorch) and [torch/cunn](https://github.com/torch/cunn);
you can install / update these by running:

```bash
luarocks install cutorch
luarocks install cunn
luarocks install cudnn
```

### (Optional) cuDNN

If you want to use NVIDIA's cuDNN library, you'll need to register for the CUDA Developer Program (it's free)
and download the library from [NVIDIA's website](https://developer.nvidia.com/cudnn); you'll also need to install
the [cuDNN bindings for Torch](https://github.com/soumith/cudnn.torch) by running

```bash
luarocks install cudnn
```

### (Optional) Training

There are some additional dependencies if you want to train your own model:

- Python 2.7
- Java JDK 1.5 or higher

You'll also need the development header files for Python 2.7 and for HDF5; you can install these
on Ubuntu by running

```bash
sudo apt-get -y install python2.7-dev
sudo apt-get install libhdf5-dev
```

You'll need the following Python libraries:
- numpy
- scipy
- Pillow
- h5py

You will also need DeepMind's [HDF5 bindings for Torch](https://github.com/deepmind/torch-hdf5) which you can install by running

```bash
luarocks install https://raw.githubusercontent.com/deepmind/torch-hdf5/master/hdf5-0-0.rockspec
```

You will need to download the pretrained VGG-16 model and the [METEOR](http://www.cs.cmu.edu/~alavie/METEOR/README.html)
evaluation code; you can do this by running the following scripts from the root directory:

```bash
sh scripts/download_models.sh
sh scripts/setup_eval.sh
```
