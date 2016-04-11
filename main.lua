--[[
Main entry point for training a DenseCap model
]]--

-------------------------------------------------------------------------------
-- Includes
-------------------------------------------------------------------------------
require 'torch'
require 'nngraph'
require 'optim'
require 'image'
require 'lfs'
require 'nn'
local cjson = require 'cjson'

require 'densecap.DataLoader'
require 'densecap.DenseCapModel'
require 'densecap.optim_updates'
local utils = require 'densecap.utils'
local opts = require 'opts'
local models = require 'models'
local eval_utils = require 'eval.eval_utils'

-------------------------------------------------------------------------------
-- Initializations
-------------------------------------------------------------------------------
local opt = opts.parse(arg)
print(opt)
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(opt.seed)
if opt.gpuid >= 0 then
  -- cuda related includes and settings
  require 'cutorch'
  require 'cunn'
  require 'cudnn'
  cutorch.manualSeed(opt.seed)
  cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed
end

-- initialize the data loader class
local loader = DataLoader(opt)
opt.seq_length = loader:getSeqLength()
opt.vocab_size = loader:getVocabSize()
opt.idx_to_token = loader.info.idx_to_token

-- initialize the DenseCap model object
local dtype = 'torch.CudaTensor'
local model = models.setup(opt):type(dtype)

-- get the parameters vector
local params, grad_params, cnn_params, cnn_grad_params = model:getParameters()
print('total number of parameters in net: ', grad_params:nElement())
print('total number of parameters in CNN: ', cnn_grad_params:nElement())
-- model.nets.lm_model:shareClones() -- TOOD: sub in single LSTM block module, get rid of this line

-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local loss_history = {}
local all_losses = {}
local results_history = {}
local iter = 0
local function lossFun()
  grad_params:zero()
  if opt.finetune_cnn_after ~= -1 and iter > opt.finetune_cnn_after then
    cnn_grad_params:zero() 
  end
  model:training()

  -- Fetch data using the loader
  local timer = torch.Timer()
  local info
  local data = {}
  data.image, data.gt_boxes, data.gt_labels, info, data.region_proposals = loader:getBatch()
  for k, v in pairs(data) do
    data[k] = v:type(dtype)
  end
  if opt.timing then cutorch.synchronize() end
  local getBatch_time = timer:time().real

  -- Run the model forward and backward
  model.timing = opt.timing
  model.cnn_backward = false
  if opt.finetune_cnn_after ~= -1 and iter > opt.finetune_cnn_after then
    model.cnn_backward = true
  end
  model.dump_vars = false
  if opt.progress_dump_every > 0 and iter % opt.progress_dump_every == 0 then
    model.dump_vars = true
  end
  local losses, stats = model:forward_backward(data)
  stats = {}
  stats.data = data
  -- stats.times['getBatch'] = getBatch_time -- this is gross but ah well

  -- Apply L2 regularization
  if opt.weight_decay > 0 then
    grad_params:add(opt.weight_decay, params)
    if cnn_grad_params then cnn_grad_params:add(opt.weight_decay, cnn_params) end
  end

  if opt.dump_all_losses == 1 then
    for k, v in pairs(losses) do
      all_losses[k] = all_losses[k] or {}
      all_losses[k][iter] = v
    end
  end
  
  --[[
  if iter % 25 == 0 then
    local boxes, scores, seq = model:forward_test(data.image)
    print(seq)
  end
  --]]
  
  --+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  -- Visualization/Logging code
  --+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  if opt.losses_log_every > 0 and iter % opt.losses_log_every == 0 then
    local losses_copy = {}
    for k, v in pairs(losses) do losses_copy[k] = v end
    loss_history[iter] = losses_copy
  end

  --[[
  if model.dump_vars then
    -- do a test-time forward pass
    -- this shouldn't affect our gradients for this iteration
    model:forward_test(data.images, {clip_final_boxes=opt.clip_final_boxes}, data.region_proposals)
    stats.vars.loss_history = loss_history
    if opt.dump_all_losses == 1 then
      stats.vars.all_losses = all_losses
    end
    stats.vars.results_history = results_history
    stats.vars.opt = opt
    if stats.vars.seq then
      -- use the dataloader to translate this to text
      stats.vars.seq = loader:decodeSequence(stats.vars.seq)
      stats.vars.target_cls = loader:decodeSequence(stats.vars.target_cls:transpose(1,2):contiguous())
    end
    -- snapshotProgress(stats.vars, opt.id)
  end
  --]]

  return losses, stats
end

-------------------------------------------------------------------------------
-- Evaluate performance of a split
-------------------------------------------------------------------------------
local function eval_split(split, max_images)
  if max_images == nil then max_images = -1 end -- -1 = disabled
  if split == nil then split = 2 end -- 2 = val, default.
  model:evaluate()
  local out = {}

  -- TODO: we're about to reset the iterator, which means that for training data
  -- we would lose our place in the dataset as we're iterating around. 
  -- This only comes up if we ever wanted to eval over training data.
  -- Worry about later.
  assert(split == 1, 'train evaluation is tricky for now. todo.')
  loader:resetIterator(split)

  -- instantiate an evaluator class
  local evaluator = DenseCaptioningEvaluator({id = opt.id})

  local counter = 0
  local all_losses = {}
  while true do
    counter = counter + 1

    -- fetch a batch of val data
    local data = {}
    data.image, data.gt_boxes, data.gt_labels, info, data.region_proposals = loader:getBatch{ split = split, iterate = true}
    for k, v in pairs(data) do
      data[k] = v:type(dtype)
    end
    local info = info[1] -- strip, since we assume only one image in batch for now

    -- evaluate the val loss function
    model.timing = false
    model.dump_vars = false
    model.cnn_backward = false
    local losses, stats = model:forward_backward(data)
    table.insert(all_losses, losses)

    -- if we are doing object detection also forward the model in test mode to get predictions and do mAP eval
    local boxes, logprobs, seq
    data.gt_boxes = data.gt_boxes[1]
    data.gt_labels = data.gt_labels[1]
    boxes, logprobs, captions = model:forward_test(data.image)

    -- seq_text = loader:decodeSequence(seq) -- translate to text
    -- target_cls_text = loader:decodeSequence(data.target_cls:transpose(1,2):contiguous())
    -- local gt_captions = loader:decodeSequence(data.gt_labels:transpose(1, 2):contiguous())
    local gt_captions = model.nets.language_model:decodeSequence(data.gt_labels)
    evaluator:addResult(logprobs, boxes, captions, data.gt_boxes, gt_captions)
  
    if boxes then
      print(string.format('processed image %s (%d/%d) of split %d, detected %d regions.',
        info.filename, info.split_bounds[1], math.min(max_images, info.split_bounds[2]), split, boxes:size(1)))
    else
      print(string.format('processed image %s (%d/%d) of split %d.',
        info.filename, info.split_bounds[1], math.min(max_images, info.split_bounds[2]), split))
    end

    -- we break out when we have processed the last image in the split bound
    if max_images > 0 and counter >= max_images then break end
    if info.split_bounds[1] == info.split_bounds[2] then break end
  end

  -- average validation loss across all images in validation data
  local loss_results = utils.dict_average(all_losses)
  print('Validation Loss Stats:')
  print(loss_results)
  print(string.format('validation loss: %f', loss_results.total_loss))
  out.loss_results = loss_results

  local ap_results = evaluator:evaluate()
  print(string.format('mAP: %f', 100*ap_results.map))
  out.ap = ap_results -- attach to output struct

  return out
end

-------------------------------------------------------------------------------
-- Main loop
-------------------------------------------------------------------------------
local loss0
local optim_state = {}
local cnn_optim_state = {}
local best_val_score = -1
while true do  

  -- Compute loss and gradient
  local losses, stats = lossFun()

  -- Parameter update
  adam(params, grad_params, opt.learning_rate, opt.optim_beta1,
       opt.optim_beta2, opt.optim_epsilon, optim_state)

  -- Make a step on the CNN if finetuning
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    adam(cnn_params, cnn_grad_params, opt.learning_rate,
         opt.optim_beta1, opt.optim_beta2, opt.optim_epsilon, cnn_optim_state)
  end

  -- print loss and timing/benchmarks
  print(string.format('iter %d: %s', iter, utils.build_loss_string(losses)))
  --print(utils.__GLOBAL_STATS__)
  if opt.timing then print(utils.build_timing_string(stats.times)) end

  if ((opt.eval_first_iteration == 1 or iter > 0) and iter % opt.save_checkpoint_every == 0) or (iter+1 == opt.max_iters) then

    -- evaluate validation performance
    local results = eval_split(1, opt.val_images_use) -- 1 = validation
    results_history[iter] = results

    -- serialize a json file that has all info except the model
    local checkpoint = {}
    checkpoint.opt = opt
    checkpoint.iter = iter
    checkpoint.loss_history = loss_history
    checkpoint.results_history = results_history
    cjson.encode_number_precision(4) -- number of sig digits to use in encoding
    cjson.encode_sparse_array(true, 2, 10)
    local text = cjson.encode(checkpoint)
    local file = io.open(opt.checkpoint_path .. '.json', 'w')
    file:write(text)
    file:close()
    print('wrote ' .. opt.checkpoint_path .. '.json')

    -- TODO: only save checkpoint if there is an improvement in mAP?
    checkpoint.model = model

    -- We want all checkpoints to be CPU compatible, so cast to float and
    -- get rid of cuDNN convolutions before saving
    model:clearState()
    model:float()
    if cudnn then
      cudnn.convert(model.net, nn)
      cudnn.convert(model.nets.localization_layer.nets.rpn, nn)
    end
    torch.save(opt.checkpoint_path, checkpoint)

    -- Now go back to CUDA and cuDNN
    model:cuda()
    if cudnn then
      cudnn.convert(model.net, cudnn)
      cudnn.convert(model.nets.localization_layer.nets.rpn, cudnn)
    end

    --[[
    local score = results.ap.map
    if score > best_val_score then
      best_val_score = score
      checkpoint.model = model
      torch.save(opt.checkpoint_path, checkpoint)
      print('wrote ' .. opt.checkpoint_path)
    end
    --]]
  end
    
  -- stopping criterions
  iter = iter + 1
  -- Collect garbage every so often
  if iter % 33 == 0 then collectgarbage() end
  if loss0 == nil then loss0 = losses.total_loss end
  if losses.total_loss > loss0 * 100 then
    model:clearState()
    torch.save('data/exploding_model.t7', model)
    torch.save('data/exploding_data.t7', stats.data)
    print('loss seems to be exploding, quitting.')
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end
end

