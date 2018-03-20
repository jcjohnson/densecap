# im2p
Tensorflow implement of paper: [A Hierarchical Approach for Generating Descriptive Image Paragraphs](http://cs.stanford.edu/people/ranjaykrishna/im2p/index.html)

I haven't fine-tunning the parameters, but I achieve the metric scores:
![metric scores](https://github.com/chenxinpeng/im2p/blob/master/img/metric_scores.png)

Please send e-mail to me if you have any questions and advices, my e-mail: xinpeng_chen@whu.edu.cn.

## Step 1
Download the [VisualGenome dataset](http://visualgenome.org/), we get the two files: VG_100K, VG_100K_2. According to the paper, we download the training, val and test splits json files. These three json files save the image names of train, validation, test data. 

Running the script:
```bash
$ python split_dataset
```
We will get images from [VisualGenome dataset] which the authors used in the paper.

##Step 2
Run the scripts:
```bash
$ python get_imgs_train_path.py
$ python get_imgs_val_path.py
$ python get_imgs_test_path.py
```
We will get three txt files: imgs_train_path.txt, imgs_val_path.txt, imgs_test_path.txt. They save the train, val, test images path.

After this, we use `dense caption` to extract features. Deploy the running environment follow by [densecap](https://github.com/jcjohnson/densecap) step by step.

Run the script:
```bash
$ ./download_pretrained_model.sh
$ th extract_features.lua -boxes_per_image 50 -max_images -1 -input_txt imgs_train_path.txt \
                          -output_h5 ./data/im2p_train_output.h5 -gpu 0 -use_cudnn 1
```
We should download the pre-trained model: `densecap-pretrained-vgg16.t7`. Then, according to the paper, we extract **50 boxes** from each image. 

Also, don't forget extract val images and test images features:
```bash
$ th extract_features.lua -boxes_per_image 50 -max_images -1 -input_txt imgs_val_path.txt \
                          -output_h5 ./data/im2p_val_output.h5 -gpu 0 -use_cudnn 1
                          
$ th extract_features.lua -boxes_per_image 50 -max_images -1 -input_txt imgs_test_path.txt \
                          -output_h5 ./data/im2p_test_output.h5 -gpu 0 -use_cudnn 1
```

## Step 3
Run the script:
```bash
$ python parse_json.py
```
In this step, we process the `paragraphs_v1.json` file for training and testing. We get the `img2paragraph` file in the **./data** directory. Its structure is like this:
![img2paragraph](https://github.com/chenxinpeng/im2p/blob/master/img/4.png)

## Step 4
Finally, we can train and test model, in the terminal:
```bash
<<<<<<< HEAD
cd webcam
virtualenv .env
source .env/bin/activate
pip install -r requirements.txt
cd ..
=======
$ CUDA_VISIBLE_DEVICES=0 ipython
>>> import HRNN_paragraph_batch.py
>>> HRNN_paragraph_batch.train()
>>>>>>> upstream2/master
```
After training, we can test the model:
```bash
>>> HRNN_paragraph_batch.test()
```

### Results
![demo](https://github.com/chenxinpeng/im2p/blob/master/img/HRNN_demo.png)
