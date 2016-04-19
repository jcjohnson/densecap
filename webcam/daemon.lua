require 'torch'
require 'nn'
require 'image'
require 'socket'

require 'densecap.DenseCapModel'

local utils = require 'densecap.utils'
local box_utils = require 'densecap.box_utils'


local cmd = torch.CmdLine()

cmd:option('-checkpoint', 'data/good_vg_checkpoints/fullcap3-475-1446663988.t7')
--cmd:option('-use_split_indicator', 1)  -- I don't think we really need this here
--cmd:option('-train_data_json', 'data/vg-regions-720-dicts.json', 'path to the json file containing additional info')
cmd:option('-max_image_size', 720)
cmd:option('-input_dir', 'webcam/inputs')
cmd:option('-input_ext', '.jpg')
cmd:option('-output_dir', 'webcam/outputs')
cmd:option('-timing', 0)

cmd:option('-rpn_nms_thresh', 0.7)
cmd:option('-final_nms_thresh', 0.7)
cmd:option('-num_proposals', 1000)
cmd:option('-gpu', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-use_cudnn', 1)
local opt = cmd:parse(arg)


local function strip_ext(path) 
  local ext = paths.extname(path) 
  return string.sub(path, 1, -#ext-2) 
end 

local function sleep(sec)
  socket.select(nil, nil, sec)
end

--[[
function decodeSequence(seq, info)
  local D,N = seq:size(1), seq:size(2)
  local out = {}
  local itow = info.idx_to_token
  for i=1,N do
    local txt = ''
    for j=1,D do
      local ix = seq[{j,i}]
      if ix >= 1 and ix <= info.vocab_size then
        -- a word, translate it
        if j >= 2 then txt = txt .. ' ' end -- space
        txt = txt .. itow[tostring(ix)]
      else
        -- END token
        break
      end
    end
    table.insert(out, txt)
  end
  return out
end

local info = utils.read_json(opt.train_data_json)
info.vocab_size = utils.count_keys(info.idx_to_token)
--]]

-- Load the checkpoint
print('loading checkpoint from ' .. opt.checkpoint)
local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model
if opt.timing == 1 then model.timing = true end
print('done loading checkpoint')

local dtype, use_cudnn = utils.setup_gpus(opt.gpu, opt.use_cudnn)
model:convert(dtype, use_cudnn)
model:setTestArgs{
  num_proposals=opt.num_proposals,
  rpn_nms_thresh=opt.rpn_nms_thresh,
  final_nms_thresh=opt.final_nms_thresh,
}

while true do
  for file in paths.files(opt.input_dir, opt.input_ext) do
    local file_id = strip_ext(file)
    local in_path = paths.concat(opt.input_dir, file)
    local out_path = paths.concat(opt.output_dir, file_id .. '.json')
    print('Running model on image ' .. in_path)

    -- Load the image
    local status, img = pcall(function() return image.load(in_path) end)

    if status then
      local ori_H, ori_W = img:size(2), img:size(3)

      -- Resize the image
      img = image.scale(img, opt.max_image_size)
      local H, W = img:size(2), img:size(3)

      -- Rescale from [0, 1] to [0, 255] and swap RGB -> BGR 
      -- and subgract vgg mean pixel
      img = img:index(1, torch.LongTensor{3, 2, 1}):mul(255)
      print(#img)

      local vgg_mean = torch.Tensor{103.939, 116.779, 123.68} -- BGR order
      img:add(-1, vgg_mean:view(3, 1, 1):expand(3, H, W))

      local timer = nil
      if opt.timing == 1 then
        cutorch.synchronize()
        timer = torch.Timer()
      end
      local boxes, scores, captions = model:forward_test(img)
        
      -- Rescale the boxes the coordinate system of the original image
      local final_boxes_xywh = box_utils.xcycwh_to_xywh(final_boxes)
      local final_boxes_xywh = box_utils.scale_boxes_xywh(final_boxes_xywh, ori_H / H)

      local output_struct = {
        boxes = final_boxes_xywh:float():totable(),
        captions = decodeSequence(seq, info),
        class_scores = class_scores:float():totable(),
        height = ori_H,
        width = ori_W,
      }

      os.remove(in_path)
      utils.write_json(out_path, output_struct)
    end
  end
  sleep(0.05)
end

