# im2p
Tensorflow implement of paper: [A Hierarchical Approach for Generating Descriptive Image Paragraphs](http://cs.stanford.edu/people/ranjaykrishna/im2p/index.html)

## Step 1
Download the [VisualGenome dataset](http://visualgenome.org/), we get the two files: VG_100K, VG_100K_2. According to the paper, we download the training, val and test splits json files. These three json files save the image names of train, validation, test data. 

Running the script:
```bash
$ python split_dataset
```
We will get images which author used in their paper.

##Step 2
