#! encoding: UTF-8

import os
import glob

train_images_path = "/home/chenxp/data/genome/im2p_train"
train_images = glob.glob(train_images_path + "/*.jpg")

f = open("imgs_train_path.txt", "w")
for item in train_images:
    f.write(item + "\n")
f.close()
