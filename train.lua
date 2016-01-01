
--[[
Main script for training a DenseCap model.
]]--

-------------------------------------------------------------------------------
-- Includes
-------------------------------------------------------------------------------
-- basics
--[[
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'image'
-- cuda related
require 'cutorch'
require 'cunn'
require 'cudnn'
-- from others
local debugger = require('fb.debugger')
-- from us
local LSTM = require 'LSTM'
local box_utils = require 'box_utils'
local utils = require 'utils'
local voc_utils = require 'voc_utils'
local cjson = require 'cjson' -- http://www.kyne.com.au/~mark/software/lua-cjson.php
require 'vis_utils'
require 'DataLoader'
require 'optim_updates'
require 'stn_detection_model'
local eval_utils = require 'eval_utils'
--]]

require 'densecap.DataLoader'
require 'densecap.DenseCapModel'
require 'densecap.optim_updates'
local utils = require 'densecap.utils'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a DenseCap model.')
cmd:text()
cmd:text('Options')

-- Core ConvNet settings
cmd:option('-backend', 'cudnn', 'nn|cudnn')

-- Model settings
cmd:option('-rpn_hidden_dim',512,'hidden size in the rpnnet')
cmd:option('-sampler_batch_size',256,'batch size to use in the box sampler')
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-input_encoding_size',512,'what is the encoding size of each token in the vocabulary? (chars or words)')
cmd:option('-sampler_high_thresh', 0.7, 'predicted boxes with IoU more than this with a positive box are considered as positive')
cmd:option('-sampler_low_thresh', 0.3, 'predicted boxes with IoU less than this with a positive box are considered as negative')
cmd:option('-train_remove_outbounds_boxes', 1,' Whether to ignore out-of-bounds boxes for sampling at training time')

-- Loss function weights
cmd:option('-mid_box_reg_weight',0.05,'what importance to assign to regressing bounding boxes well in rpn?')
cmd:option('-mid_objectness_weight', 0.1, 'what importance to assign to pos/neg objectness labels?')
cmd:option('-end_box_reg_weight', 0.1, 'what importance to assign to final class-specific bounding box regression?')
cmd:option('-end_objectness_weight',0.1,'what importance to assign to classifying the correct class?')
cmd:option('-captioning_weight',1.0,'what importance to assign to captioning, if present?')
cmd:option('-weight_decay', 1e-6, 'L2 weight decay penalty strength')
cmd:option('-box_reg_decay', 5e-5, 'Strength of a pull that boxes experience towards their anchor, to prevent wild drifts')

-- Data input settings
cmd:option('-train_data_h5','data/VG-regions.h5','path to the h5file containing the preprocessed dataset (made in prepro.py)')
cmd:option('-train_data_json','data/VG-regions-dicts.json','path to the json file containing additional info (made in prepro.py)')
cmd:option('-h5_read_all',false,'read the whole h5 dataset to memory? COCO images take several tens of GB might not fit in your RAM, need partial reading.')
cmd:option('-proposal_regions_h5','','override RPN boxes with boxes from this h5 file (empty = don\'t override)')
cmd:option('-debug_max_train_images', -1, 'for debugging: Cap #train images at this value to check that we can overfit. (-1 = disable)')

-- Optimization
cmd:option('-learning_rate',4e-6,'learning rate to use')
cmd:option('-optim_beta1',0.9,'beta1 for adam')
cmd:option('-optim_beta2',0.999,'beta2 for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon for smoothing')
cmd:option('-drop_prob', 0.5, 'Dropout strength throughout the model.')
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-checkpoint_start_from', '', 'load model from a given checkpoint instead of random initialization.')
cmd:option('-finetune_cnn_after', -1, 'After what iteration do we start finetuning the CNN? (-1 = disable; never finetune, 0 = finetune from start)')
cmd:option('-val_images_use', 100, 'how many images to use when periodically evaluating the validation loss? (-1 = all)')

-- Model checkpointing
cmd:option('-save_checkpoint_every', 1000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'checkpoint.t7', 'the name of the checkpoint file to use')

-- Visualization
cmd:option('-progress_dump_every', 100, 'Every how many iterations do we write a progress report to vis/out ?. 0 = disable.')
cmd:option('-losses_log_every', 10, 'How often do we save losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-id', '', 'an id identifying this run/job. can be used in cross-validation.')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-timing', false, 'whether to time parts of the net')
cmd:option('-dump_all_losses', 0)
cmd:option('-clip_final_boxes', 1,
           'whether to clip final boxes to image boundary; probably set to 0 for dense captioning')
cmd:option('-eval_first_iteration',0,'evaluate on first iteration? 1 = do, 0 = dont.')

cmd:text()

-------------------------------------------------------------------------------
-- Initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)
cutorch.setDevice(opt.gpuid + 1) -- note +1 because lua is 1-indexed

-- initialize the data loader class
local dataOpt = {}
dataOpt.h5_file = opt.train_data_h5
dataOpt.json_file = opt.train_data_json
dataOpt.h5_read_all = opt.h5_read_all
dataOpt.debug_max_train_images = opt.debug_max_train_images
dataOpt.proposal_regions_h5 = opt.proposal_regions_h5
local loader = DataLoader(dataOpt)
local seq_length = loader:getSeqLength()
local max_image_size = loader:getImageMaxSize()
local vocab_size = loader:getVocabSize()

local model
if opt.checkpoint_start_from == '' then
  opt.vocab_size = vocab_size -- hmm, not very pretty...
  opt.seq_length = seq_length
  model = DenseCapModel(opt, loader.info.idx_to_token)
else
  print('initializing model from ' .. opt.checkpoint_start_from)
  model = torch.load(opt.checkpoint_start_from).model
  print(model)
  model.opt.objectness_weight = opt.objectness_weight
  model.nets.detection_module.opt.obj_weight = opt.objectness_weight
  model.opt.box_reg_weight = opt.box_reg_weight
  model.nets.box_reg_crit.w = opt.final_box_reg_weight
  model.opt.classification_weight = opt.classification_weight
  local rpn = model.nets.detection_module.nets.rpn
  rpn:findModules('nn.RegularizeLayer')[1].w = opt.box_reg_decay
  model.opt.sampler_high_thresh = opt.iou_high_thresh
  model.opt.sampler_low_thresh = opt.iou_low_thresh
  model.opt.train_remove_outbounds_boxes = opt.train_remove_outbounds_boxes
  model.opt.captioning_weight = opt.captioning_weight
end

-- Find all Dropout layers and set their probabilities according to provided option
local dropout_modules = model.nets.recog_base:findModules('nn.Dropout')
for i, dropout_module in ipairs(dropout_modules) do
  dropout_module.p = opt.drop_prob
end

-- get the parameters vector
local params, grad_params, cnn_params, cnn_grad_params = model:getParametersSeparate{
                              with_cnn=opt.finetune_cnn_after == 0,
                              with_rpn=true,
                              with_recog_net=true}
print('total number of parameters in net: ', grad_params:nElement())
if cnn_grad_params then
  print('total number of parameters in CNN: ', cnn_grad_params:nElement())
else
  print('CNN is not being trained right now (0 parameters).')
end

if model.usernn then
  model.nets.lm_model:shareClones()
end

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
  data.images, data.target_boxes, data.target_seqs, info, data.region_proposals = loader:getBatch()
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
  stats.times['getBatch'] = getBatch_time -- this is gross but ah well

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

  if iter % 25 == 0 then
    local boxes, scores, seq = model:forward_test(data.images)
    local txt = model:decodeSequence(seq)
    print(txt)
  end
  
  --+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  -- Visualization/Logging code
  --+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  if opt.losses_log_every > 0 and iter % opt.losses_log_every == 0 then
    local losses_copy = {}
    for k, v in pairs(losses) do losses_copy[k] = v end
    loss_history[iter] = losses_copy
  end

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
  -- This only comes up if we ever wanted to eval over training data. Worry about later.
  assert(split == 1, 'train evaluation is tricky for now. todo.')
  loader:resetIterator(split)

  -- instantiate an evaluator class
  local evaluator = DenseCaptionEvaluator({id = opt.id})

  local counter = 0
  local all_losses = {}
  while true do
    counter = counter + 1

    -- fetch a batch of val data
    local data = {}
    data.images, data.target_boxes, data.target_cls, info, data.region_proposals = loader:getBatch{ split = split, iterate = true}
    local info = info[1] -- strip, since we assume only one image in batch for now

    -- evaluate the val loss function
    model.timing = false
    model.dump_vars = false
    model.cnn_backward = false
    local losses, stats = model:forward_backward(data)
    table.insert(all_losses, losses)

    -- if we are doing object detection also forward the model in test mode to get predictions and do mAP eval
    local boxes, logprobs, seq
    data.target_boxes = data.target_boxes[1]
    data.target_cls = data.target_cls[1]
    boxes, logprobs, seq = model:forward_test(data.images, {clip_final_boxes=opt.clip_final_boxes}, data.region_proposals)

    seq_text = loader:decodeSequence(seq) -- translate to text
    target_cls_text = loader:decodeSequence(data.target_cls:transpose(1,2):contiguous())
    evaluator:addResult(info, boxes, logprobs, data.target_boxes, target_cls_text, seq_text)
  
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

  -- We need to rebuild params and grad_params when we start finetuning
  if iter > 0 and iter == opt.finetune_cnn_after then
    params, grad_params, cnn_params, cnn_grad_params = model:getParametersSeparate{
                            with_cnn=true,
                            with_rpn=true,
                            with_pred_net=true
                         }
    print('adding CNN to the optimization.')
    print('total number of parameters in net: ', grad_params:nElement())
    print('total number of parameters in CNN: ', cnn_params:nElement())
    if model.usernn then -- this must be here because memory was reallocated around with :getParameters(). tricky!
      model.nets.lm_model:shareClones()
    end
  end

  -- compute the gradient
  local losses, stats = lossFun()

  -- perform parameter update on the model
  adam(params, grad_params, opt.learning_rate, opt.optim_beta1, opt.optim_beta2, opt.optim_epsilon, optim_state)
  -- maybe also perform parameter update on the CNN
  if opt.finetune_cnn_after >= 0 and iter >= opt.finetune_cnn_after then
    adam(cnn_params, cnn_grad_params, opt.learning_rate, opt.optim_beta1, opt.optim_beta2, opt.optim_epsilon, cnn_optim_state)
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

    -- add the model and save it (only if there was improvement in map)
    local score = results.ap.map
    if score > best_val_score then
      best_val_score = score
      checkpoint.model = model
      torch.save(opt.checkpoint_path, checkpoint)
      print('wrote ' .. opt.checkpoint_path)
    end
  end
    
  -- stopping criterions
  iter = iter + 1
  if iter % 33 == 0 then collectgarbage() end -- good idea to do this once in a while
  if loss0 == nil then loss0 = losses.total_loss end
  if losses.total_loss > loss0 * 100 then
    print('loss seems to be exploding, quitting.')
    break
  end
  if opt.max_iters > 0 and iter >= opt.max_iters then break end -- stopping criterion
end

