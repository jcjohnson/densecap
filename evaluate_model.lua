require 'torch'
require 'nn'

require 'densecap.DataLoader'
require 'densecap.DenseCapModel'

local eval_utils = require 'eval.eval_utils'

--[[
Evaluate a trained DenseCap model by running it on a split on the data.
--]]

local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'data/checkpoint.t7', 'The checkpoint to evaluate')
cmd:option('-data_h5', '', 'The HDF5 file to load data from; optional.')
cmd:option('-data_json', '' 'The JSON file to load data from; optional.')
cmd:option('-gpu', 0, 'The GPU to use; set to -1 for CPU')
cmd:option('-use_cudnn', 1, 'Whether to use cuDNN backend in GPU mode.')
cmd:option('-split', 'val', 'Which split to evaluate; either val or test.')
cmd:option('-max_images', -1 'How many images to evaluate; -1 for whole split')
local opt = cmd:parse(arg)


-- First load the model
local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model
print 'Loaded model'

-- Figure out CPU / GPU and cudnn
local dtype = 'torch.FloatTensor'
local use_cudnn = false
if opt.gpu >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.use_cudnn == 1 then
    require 'cudnn'
    use_cudnn = true
  end
  dtype = 'torch.CudaTensor'
  cutorch.setDevice(opt.gpu + 1)
end
print(string.format('Using dtype "%s"', dtype))

-- Cast the model to the right dtype and convert convolutions from nn to cudnn
model:type(dtype)
if use_cudnn then
  cudnn.convert(model.net, cudnn)
  cudnn.convert(model.nets.localization_layer.nets.rpn, cudnn)
end

-- Set up the DataLoader; use HDF5 and JSON files from checkpoint if they were
-- not explicitly provided.
if opt.data_h5 == '' then
  opt.data_h5 = checkpoint.opt.data_h5
end
if opt.data_json == '' then
  opt.data_json = checkpoint.opt.data_json
end
local loader = DataLoader(opt)

-- Actually run evaluation
local eval_kwargs = {
  model=model,
  loader=loader,
  split=opt.split,
  max_images=opt.max_images,
  dtype=dtype,
}
local eval_results = eval_utils.eval_split(eval_kwargs)
