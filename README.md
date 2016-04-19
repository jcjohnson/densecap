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

## Running on new images

To run the model on new images, use the script `run_model.lua`. To run the pretrained model on the provided `elephant.jpg` image,
use the following command:

```bash
th run_model.lua -input_image elephant.jpg
```

By default this will run in GPU mode; to run in CPU only mode, simply add the flag `-gpu -1`.

This command will write results into the folder `vis/data`. We have provided a web-based visualizer to view these
results; to use it, change to the `vis` directory and start a local HTTP server:

```bash
cd vis
python -m SimpleHTTPServer 8181
```

Then point your web browser to [http://localhost:8181/view_results.html](http://localhost:8181/view_results.html).

If you have an entire directory of images on which you want to run the model, use the `-input_dir` flag instead:

```bash
th run_model.lua -input_dir /path/to/my/image/folder
```

This run the model on all files in the folder `/path/to/my/image/folder/` whose filename does not start with `.`.

The web-based visualizer is the prefered way to view results, but if you don't want to use it then you can instead
render an image with the detection boxes and captions "baked in"; add the flag `-output_dir` to specify a directory
where output images should be written:

```bash
th run_model.lua -input_dir /path/to/my/image/folder -output_dir /path/to/output/folder/
```
