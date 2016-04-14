require 'torch'
require 'nn'
require 'image'

require 'densecap.DenseCapModel'
local utils = require 'densecap.utils'
local box_utils = require 'densecap.box_utils'
local vis_utils = require 'densecap.vis_utils'


--[[
Run a trained DenseCap model on images.

The inputs can be any one of:
- a single image: use the flag '-input_image' to give path
- a directory with images: use flag '-input_dir' to give dir path
- MSCOCO split: use flag '-input_split' to identify the split (train|val|test)

The output can be controlled with:
- max_images: maximum number of images to process. Set to -1 to process all
- output_dir: use this flag to identify directory to write outputs to
- output_vis: set to 1 to output images/json to the vis directory for nice viewing in JS/HTML

TODO:
- Add options to better configure test-time behavior:
  - Number of region proposals
  - NMS?
- Actually save output as JSON
--]]


local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'data/checkpoint.t7')
cmd:option('-image_size', 720)
-- input settings
cmd:option('-input_image', '', 'A path to a single specific image to caption')
cmd:option('-input_dir', '', 'A path to a directory with images to caption')
cmd:option('-input_split', '', 'An MSCOCO split identifier to process (train|val|test)')
-- output settings
cmd:option('-max_images', 100, 'max number of images to process')
cmd:option('-output_dir', '')
  -- these settings are only used if output_dir is not empty
  cmd:option('-num_to_draw', 10, 'max number of predictions per image')
  cmd:option('-text_size', 1, '1 looks best I think')
  cmd:option('-box_width', 2, 'width of rendered box')
cmd:option('-output_vis', 1)
-- misc
cmd:option('-gpu', 0)
cmd:option('-use_cudnn', 1)
local opt = cmd:parse(arg)
print(opt)

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

  -- Run the model forward
  local boxes, scores, captions = model:forward_test(img_caffe:type(dtype))
  local boxes_xywh = box_utils.xcycwh_to_xywh(boxes)

  local out = {
    img = img,
    boxes = boxes_xywh,
    scores = scores,
    captions = captions,
  }
  return out
end


function lua_render_result(result, opt)
  -- use lua utilities to render results onto the image (without going)
  -- through the vis utilities written in JS/HTML. Kind of ugly output.

  -- respect the num_to_draw setting and slice the results appropriately
  local boxes = result.boxes
  local num_boxes = math.min(opt.num_to_draw, boxes:size(1))
  boxes = boxes[{{1, num_boxes}}]
  local captions_sliced = {}
  for i = 1, num_boxes do
    table.insert(captions_sliced, result.captions[i])
  end

  -- Convert boxes and draw output image
  local draw_opt = { text_size = opt.text_size, box_width = opt.box_width }
  local img_out = vis_utils.densecap_draw(result.img, boxes, captions_sliced, draw_opt)
  return img_out
end

function get_input_images(opt)
  -- utility function that figures out which images we should process 
  -- and fetches all the raw image paths
  local image_paths = {}
  if opt.input_image ~= '' then
    table.insert(image_paths, opt.input_image)
  elseif opt.input_dir ~= '' then
    -- iterate all files in input directory and add them to work
    for fn in paths.files(opt.input_dir) do
      if string.sub(fn, 1, 1) ~= '.' then
        local img_in_path = paths.concat(opt.input_dir, fn)
        table.insert(image_paths, img_in_path)
      end
    end
  end
  return image_paths
end

-- Load the model, and cast to the right type
local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model
model:type(dtype)
if use_cudnn then 
  cudnn.convert(model.net, cudnn)
end
model:evaluate()

-- get paths to all images we should be evaluating
local image_paths = get_input_images(opt)
local num_process = math.min(#image_paths, opt.max_images)
for k=1,num_process do
  local img_path = image_paths[k]
  print(string.format('%d/%d processing image %s', k, num_process, img_path))
  -- run the model on the image and obtain results
  local result = run_image(model, img_path, opt, dtype)  

  if opt.output_dir ~= '' then
    local img_out = lua_render_result(result, opt)
    local img_out_path = paths.concat(opt.output_dir, paths.basename(img_path))
    image.save(img_out_path, img_out)
  end

end

