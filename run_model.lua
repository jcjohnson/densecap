require 'torch'
require 'nn'
require 'image'

require 'densecap.DenseCapModel'
local utils = require 'densecap.utils'
local box_utils = require 'densecap.box_utils'
local vis_utils = require 'densecap.vis_utils'


--[[
Run a trained DenseCap model on raw images, outputting either images with boxes
and captions drawn into the image or a JSON file with annotations for downstream
processing.

TODO:
- Add options to better configure test-time behavior:
  - Number of region proposals
  - NMS?
- Actually save output as JSON
--]]


local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'data/checkpoint.t7')
cmd:option('-input_image', '')
cmd:option('-input_dir', '')

cmd:option('-output_dir', 'outputs')
cmd:option('-save_json', 1)
cmd:option('-save_images', 1)

cmd:option('-image_size', 720)
cmd:option('-num_to_draw', 10)
cmd:option('-text_size', 1)
cmd:option('-box_width', 2)

cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 1)
local opt = cmd:parse(arg)


function run_image(model, img_path, opt, dtype)
  -- Load, resize, and preprocess image
  local img = image.load(img_path, 3)
  img = image.scale(img, opt.image_size):float()
  local H, W = img:size(2), img:size(3)
  local img_caffe = img:view(1, 3, H, W)
  img_caffe = img_caffe:index(2, torch.LongTensor{3, 2, 1}):mul(255)
  local vgg_mean = torch.FloatTensor{103.939, 116.779, 123.68}
  vgg_mean = vgg_mean:view(1, 3, 1, 1):expand(1, 3, H, W)
  img_caffe:add(-1, vgg_mean)

  -- Run model, and keep only the top detections
  local boxes, scores, captions = model:forward_test(img_caffe:type(dtype))
  local num_boxes = math.min(opt.num_to_draw, boxes:size(1))
  boxes = boxes[{{1, num_boxes}}]
  scores = scores[{{1, num_boxes}}]
  local captions_sliced = {}
  for i = 1, num_boxes do
    table.insert(captions_sliced, captions[i])
  end

  captions = captions_sliced

  -- Convert boxes and draw output image
  local boxes_xywh = box_utils.xcycwh_to_xywh(boxes)
  local draw_opt = {
    text_size = opt.text_size,
    box_width = opt.box_width,
  }
  local img_out = vis_utils.densecap_draw(img, boxes_xywh, captions, draw_opt)

  local json_struct = {
    boxes = boxes_xywh,
    captions = captions,
  }

  return img_out, json_struct
end

-- Figure out datatypes
local dtype = 'torch.FloatTensor'
local use_cudnn = false
if opt.gpu >= 0 then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  dtype = 'torch.CudaTensor'
  if opt.use_cudnn == 1 then
    require 'cudnn'
    use_cudnn = true
  end
end

-- Load the model, and cast to the right type
local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model
model:type(dtype)
if use_cudnn then
  cudnn.convert(model.net, cudnn)
end
model:evaluate()

if opt.input_image ~= '' then
  local img_out, json_struct = run_image(model, opt.input_image, opt, dtype)
  if opt.save_images == 1 then
    local img_out_path = paths.concat(opt.output_dir, paths.basename(opt.input_image))
    image.save(img_out_path, img_out)
  end
  if opt.save_json == 1 then
    print 'TODO: ACTUALLY SAVE JSON'
  end
end

if opt.input_dir ~= '' then
  for fn in paths.files(opt.input_dir) do
    if string.sub(fn, 1, 1) ~= '.' then
      local img_in_path = paths.concat(opt.input_dir, fn)
      local img_out, json_struct = run_image(model, img_in_path, opt, dtype)
      if opt.save_images == 1 then
        local img_out_path = paths.concat(opt.output_dir, fn)
        image.save(img_out_path, img_out)
      end
      if opt.save_json == 1 then
        print 'TODO: ACTUALLY SAVE JSON'
      end
    end
  end
end
