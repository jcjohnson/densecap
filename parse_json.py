# encoding: UTF-8

import os
import json
import cPickle

paragraph_json_file = open('./data/paragraphs_v1.json').read()
paragraph = json.loads(paragraph_json_file)

img2paragraph = {}

for each_img in paragraph:
    image_id = each_img['image_id']
    each_paragraph = each_img['paragraph']
    sentences = each_paragraph.split('. ')
    if '' in sentences:
        sentences.remove('')
    img2paragraph[image_id] = [len(sentences), sentences]

for key, para in img2paragraph.iteritems():
    print key, para

with open('img2paragraph', 'wb') as f:
    cPickle.dump(img2paragraph, f)
