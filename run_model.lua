require 'torch'
require 'nn'
require 'image'

require 'cunn'
require 'cudnn'

require 'densecap.DenseCapModel'
local utils = require 'densecap.utils'
local box_utils = require 'densecap.box_utils'
local vis_utils = require 'densecap.vis_utils'

local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'data/checkpoint.t7')
cmd:option('-input_image', 'train20/2375025.jpg')
cmd:option('-output_image', 'foo.jpg')
cmd:option('-image_size', 720)
cmd:option('-num_to_draw', 10)
local opt = cmd:parse(arg)


local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model
model:evaluate()
local img = image.load(opt.input_image, 3)
img = image.scale(img, opt.image_size):float()

local H, W = img:size(2), img:size(3)
local img_caffe = img:view(1, 3, H, W)
img_caffe = img_caffe:index(2, torch.LongTensor{3, 2, 1}):mul(255)
local vgg_mean = torch.FloatTensor{103.939, 116.779, 123.68}
vgg_mean = vgg_mean:view(1, 3, 1, 1)
img_caffe:add(-1, vgg_mean:expand(1, 3, H, W))

local boxes, scores, captions = model:forward_test(img_caffe:cuda())
boxes = boxes[{{1, opt.num_to_draw}}]
scores = scores[{{1, opt.num_to_draw}}]
local captions_sliced = {}
for i = 1, opt.num_to_draw do
  table.insert(captions_sliced, captions[i])
end
captions = captions_sliced

local boxes_xywh = box_utils.xcycwh_to_xywh(boxes)
local img_out = vis_utils.densecap_draw(img, boxes_xywh, captions)
image.save(opt.output_image, img_out)
