require 'torch'
require 'nn'

require 'densecap.ApplyBoxTransform'
require 'densecap.BilinearRoiPooling'
require 'densecap.BoxRegressionCriterion'
require 'densecap.DetectionModule'
require 'densecap.LogisticCriterion'
require 'densecap.LanguageModel'

local box_utils = require 'densecap.box_utils'
local net_utils = require 'densecap.net_utils'
local utils = require 'densecap.utils'

local DenseCapModel = torch.class('DenseCapModel')

--------------------------------------------------------------------------------

function DenseCapModel:__init(opt, idx_to_token)
  opt = opt or {}  
  opt.cnn_name = utils.getopt(opt, 'cnn_name', 'vgg-16')
  opt.backend = utils.getopt(opt, 'backend', 'cudnn')
  opt.path_offset = utils.getopt(opt, 'path_offset', '')
  opt.dtype = utils.getopt(opt, 'dtype', 'torch.CudaTensor')
  opt.vocab_size = utils.getopt(opt, 'vocab_size')
  opt.std = utils.getopt(opt, 'std', 0.01) -- Used to initialize new layers
  
  assert(idx_to_token)
  print('saving idx_to_token')
  self.idx_to_token = idx_to_token

  -- Ensure that all options for loss were specified
  utils.ensureopt(opt, 'mid_box_reg_weight')
  utils.ensureopt(opt, 'mid_objectness_weight')
  utils.ensureopt(opt, 'end_box_reg_weight')
  utils.ensureopt(opt, 'end_objectness_weight')
  utils.ensureopt(opt, 'captioning_weight')
  
  -- Options for RNN
  opt.seq_length = utils.getopt(opt, 'seq_length')
  opt.rnn_encoding_size = utils.getopt(opt, 'rnn_encoding_size', 512)
  opt.rnn_size = utils.getopt(opt, 'rnn_size', 512)
  self.opt = opt

  -- This will hold all components of the model.
  self.nets = {}
  
  -- Load the CNN from disk
  local cnn = net_utils.load_cnn(opt.cnn_name, opt.backend, opt.path_offset)
  
  -- We need to chop the CNN into three parts: conv that is not finetuned,
  -- conv that will be finetuned, and fully-connected layers. We'll just
  -- hardcode the indices of these layers per architecture.
  local conv_start1, conv_end1, conv_start2, conv_end2
  local recog_start, recog_end
  local fc_dim
  if opt.cnn_name == 'vgg-16' then
    conv_start1, conv_end1 = 1, 10 -- these will not be finetuned for efficiency
    conv_start2, conv_end2 = 11, 30 -- these will be finetuned possibly
    recog_start, recog_end = 32, 38 -- FC layers
    opt.input_dim = 512
    opt.output_height, opt.output_width = 7, 7
    fc_dim = 4096
  else
    error(string.format('Unrecognized CNN "%s"', opt.cnn_name))
  end

  -- Now that we have the indices, actually chop up the CNN.
  self.nets.conv_net1 = net_utils.subsequence(cnn, conv_start1, conv_end1)
  self.nets.conv_net2 = net_utils.subsequence(cnn, conv_start2, conv_end2)

  -- 
  local conv_full = net_utils.subsequence(cnn, conv_start1, conv_end2)
  local x0, y0, sx, sy = net_utils.compute_field_centers(conv_full)
  self.opt.field_centers = {x0, y0, sx, sy}
  self.nets.detection_module = nn.DetectionModule(opt)
  -- self.nets.fixed_detection_module = nn.FixedDetectionModule(opt)

  -- recognition base network (e.g. FC layers from VGG). Terminates with fc_dim dimension codes
  self.nets.recog_base = net_utils.subsequence(cnn, recog_start, recog_end)
  -- create a classification (or objectness) network. Output is (N,C) for detection or (N,2) in captioning (binary)
  -- in case of classification this is, e.g. 20 outputs (classes). In captioning it's just binary 0/1

  local class_layer = nn.Linear(fc_dim, 2)
  class_layer.weight:normal(0, opt.std)
  class_layer.bias:zero()
  self.nets.recog_class = class_layer
  -- create box regression network. Its output is (N/2,C,4) in detection or (N/2,4) in captioning
  -- its input is also N/2 because we only run this network on the positive boxes. (of course N/2 is approximate)
  -- in classification we use class-specific regression. In captioning it's just one box correction

  local box_reg_branch = nn.Sequential()
  local box_reg_layer = nn.Linear(fc_dim, 4)
  box_reg_layer.weight:zero()
  box_reg_layer.bias:zero()
  box_reg_branch:add(box_reg_layer)

  self.nets.recog_box = box_reg_branch
  
  -- LM encoder, e.g. mapping the image features from 4096 -> 512, to be plugged into the RNN
  self.nets.recog_lm_encode = nn.Sequential()
  self.nets.recog_lm_encode:add(nn.Linear(fc_dim, opt.rnn_encoding_size))
  self.nets.recog_lm_encode:add(nn.ReLU()) -- todo maybe use backend here instead of hardcoding nn
  -- create the Language Model
  local lmOpt = {}
  lmOpt.vocab_size = opt.vocab_size
  lmOpt.input_encoding_size = opt.rnn_encoding_size
  lmOpt.rnn_size = opt.rnn_size
  lmOpt.seq_length = opt.seq_length
  self.nets.lm_model = nn.LanguageModel(lmOpt)
  self.nets.lm_crit = nn.LanguageModelCriterion()

  -- Set up criterions for final objectness and box regression
  self.nets.class_crit = nn.LogisticCriterion()
  self.nets.box_reg_crit = nn.BoxRegressionCriterion(opt.end_box_reg_weight)

  -- Used at test-time for final bounding box regression
  self.nets.apply_box_transform = nn.ApplyBoxTransform()

  -- Intermediate buffers used in forward / backward
  self.grad_roi_boxes = torch.Tensor()

  -- Cast everything to the right datatype
  for k, v in pairs(self.nets) do
    self.nets[k]:type(self.opt.dtype)
  end

  self.timer = torch.Timer()
  self:reset_stats()

  self.timing = false
  self.dump_vars = false
  self.cnn_backward = false
end


function DenseCapModel:timeit(name, f)
  self.timer = self.timer or torch.Timer()
  if self.timing then
    cutorch.synchronize()
    self.timer:reset()
    f()
    cutorch.synchronize()
    self.stats.times[name] = self.timer:time().real
  else
    f()
  end
end


-- Set all sub-modules to training mode
function DenseCapModel:training()
  for k, v in pairs(self.nets) do
    if v.training then
      v:training()
    end
  end
end


-- Set all sub-modules to evaluation mode
function DenseCapModel:evaluate()
  for k, v in pairs(self.nets) do
    if v.evaluate then
      v:evaluate()
    end
  end
end


function DenseCapModel:reset_stats()
  self.stats = {}
  self.stats.times = {}
  self.stats.losses = {}
  self.stats.vars = {}
end


--[[
Decode a LongTensor of token indices into a table of strings.

Input:
- seq: LongTensor of shape D x N where each element is between 1 and self.opt.vocab_size.

Returns:
- captions: An array with N strings, each of length < D.
--]]
function DenseCapModel:decodeSequence(seq)
  local D, N = seq:size(1), seq:size(2)
  local out = {}
  local itow = self.idx_to_token
  for i = 1, N do
    local txt = ''
    for j = 1, D do
      local ix = seq[{j,i}]
      if ix >= 1 and ix <= self.opt.vocab_size then
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


--[[
Run the model forward to compute loss and backward to compute gradients.

Inputs:
  - data: Table with the following keys:
    - images: N x 3 x H x W array of pixel data
    - target_boxes: N x B2 x 4 array of ground-truth object boxes
    - target_seqs: N x B2 x 1 array of ground-truth sequences for objects
    - region_proposals: N x B3 x (4 or 5) array of region proposals in (xc, yc, w, h) format.
      If these are passed then they will be used instead of using RPN region proposals.
    - cnn_features: For debugging; use these instead of actually running the CNN forward
--]]
function DenseCapModel:forward_backward(data)
  self:training()

  -- Cast input data to correct type
  for k, v in pairs(data) do
    data[k] = v:type(self.opt.dtype)
  end
  
  local region_proposals = data.region_proposals
  if region_proposals then
    -- If region proposals were given, make sure they have the right shape
    assert(region_proposals:dim() == 3 and region_proposals:size(1) == 1)
    local last_dim = region_proposals:size(3)
    assert(last_dim == 4 or last_dim == 5)
    if last_dim == 5 then
      -- If input region proposals had 5 columns, slice out the last one.
      region_proposals = region_proposals[{{}, {}, {1, 4}}]:contiguous()
    end
  end

  -- Run cnn forward
  local cnn_features, cnn_features1
  if data.cnn_features then
    cnn_features = data.cnn_features
  else
    self:timeit('cnn:forward', function()
      cnn_features1 = self.nets.conv_net1:forward(data.images)
      cnn_features = self.nets.conv_net2:forward(cnn_features1)
    end)
  end

  -- Run DetectionModule forward
  local image_height, image_width = data.images:size(3), data.images:size(4)

  local roi_features, roi_boxes, pos_gt_boxes, pos_gt_labels
  if region_proposals then
    -- Use the provided region proposals, so run FixedDetectionModule forward
    self.nets.fixed_detection_module.timing = self.timing
    self.nets.fixed_detection_module:setImageSize(image_height, image_width)
    local det_out = self.nets.fixed_detection_module:forward{
                      cnn_features, region_proposals, data.target_boxes, data.target_seqs
                    }
    roi_features, roi_boxes, pos_gt_boxes, pos_gt_labels = unpack(det_out)
  else
    self.nets.detection_module.timing = self.timing
    self.nets.detection_module.dump_vars = self.dump_vars
    self.nets.detection_module:setImageSize(image_height, image_width)
    local det_out = self.nets.detection_module:forward{
                      cnn_features, data.target_boxes, data.target_seqs
                    }
    roi_features, roi_boxes, pos_gt_boxes, pos_gt_labels = unpack(det_out)
  end
  local num_pos = pos_gt_boxes:size(1)
  local num_neg = roi_features:size(1) - num_pos
  
  -- Copy over all losses from the DetectionModule
  if not region_proposals then
    for k, v in pairs(self.nets.detection_module.stats.losses) do
      if k ~= 'total_loss' then
        self.stats.losses[k] = v
      end
    end
  end

  -- Run the base recognition network
  local roi_codes = self.nets.recog_base:forward(roi_features) -- e.g. returns (N,4096) matrix output from VGG FC layers
  local roi_codes_pos = roi_codes[{{1,num_pos}}]

  -------------------------------------------------------------------------------
  -- Process the classification loss
  -------------------------------------------------------------------------------
  local class_scores = self.nets.recog_class:forward(roi_codes)
  
  -- Construct the labels vector
  local gt_labels_vec = torch.LongTensor(num_pos + num_neg):zero() -- class/objectness labels vector
  local pos_gt_labels_vec
  gt_labels_vec[{{1, num_pos}}]:fill(1)

  -- Compute classification loss and gradient for positives and negatives
  local crit = self.nets.class_crit
  local weight = self.opt.end_objectness_weight
  local loss = crit:forward(class_scores, gt_labels_vec)
  self.stats.losses.class_loss = weight * loss
  local grad_class_scores = crit:backward(class_scores, gt_labels_vec)
  grad_class_scores:mul(weight)
  -- lets do inline backward pass for classification loss as well
  local grad_roi_codes = self.nets.recog_class:backward(roi_codes, grad_class_scores)
  -------------------------------------------------------------------------------
  -- Process the box regression loss (only for positives)
  -------------------------------------------------------------------------------
  local pos_class_trans = self.nets.recog_box:forward(roi_codes_pos)
  local pos_trans_gathered = pos_class_trans -- noop in case of captioning

  -- Compute box regression loss and gradient
  local crit = self.nets.box_reg_crit
  local pos_roi_boxes = roi_boxes[{{1, num_pos}}]
  local loss = crit:forward({pos_roi_boxes, pos_trans_gathered}, pos_gt_boxes)
  self.stats.losses.final_box_reg_loss = loss
  local din = crit:backward({pos_roi_boxes, pos_trans_gathered}, pos_gt_boxes)
  local grad_pos_roi_boxes, grad_pos_trans_gathered = unpack(din)
  -- do inline backward pass for regression loss
  local grad_pos_class_trans = grad_pos_trans_gathered

  local grad_roi_codes_pos = self.nets.recog_box:backward(roi_codes_pos, grad_pos_class_trans)
  -- accumulate the gradient into grad_roi_codes in the proper place. Note: we are reusing the memory that is owned by nets.recog_class
  grad_roi_codes[{{1,num_pos}}]:add(grad_roi_codes_pos)
  -- expand the gradient for roi_boxes into proper dimensions. Using instance method to avoid reallocation of memory
  self.grad_roi_boxes = self.grad_roi_boxes:typeAs(roi_boxes)
  self.grad_roi_boxes:resizeAs(roi_boxes):zero()
  self.grad_roi_boxes[{{1, num_pos}}]:copy(grad_pos_roi_boxes)
  -------------------------------------------------------------------------------
  -- Process the language model loss (in case of captioning, and only for positives)
  -------------------------------------------------------------------------------
  -- encode the roi_codes before plugging into the LM
  local roi_codes_pos_encoded = self.nets.recog_lm_encode:forward(roi_codes_pos)
  -- Language model wants labels transposed for efficiency
  local seq_transposed = pos_gt_labels:transpose(1,2):contiguous()
  local lm_scores = self.nets.lm_model:forward{roi_codes_pos_encoded, seq_transposed}
  -- process the criterion and the loss
  local crit = self.nets.lm_crit
  local weight = self.opt.captioning_weight
  local loss = crit:forward(lm_scores, seq_transposed)
  self.stats.losses.captioning_loss = weight * loss
  grad_lm_scores = crit:backward(lm_scores, seq_transposed)
  grad_lm_scores:mul(weight)
  -- backprop the lm model
  local grad_roi_codes_pos_encoded = self.nets.lm_model:backward({roi_codes_pos_encoded, seq_transposed}, grad_lm_scores)[1]
  -- backprop the encoding
  local grad_roi_codes_pos = self.nets.recog_lm_encode:backward(roi_codes_pos, grad_roi_codes_pos_encoded)
  -- accumulate the gradient into grad_roi_codes in the proper place. Note: we are reusing the memory that is owned by nets.recog_class
  grad_roi_codes[{{1,num_pos}}]:add(grad_roi_codes_pos)
--------------------------------------------------------------------
  -- Rest of backward pass
  -------------------------------------------------------------------------------
  -- backprop the recognition base network
  local grad_roi_features = self.nets.recog_base:backward(roi_features, grad_roi_codes)
  -- backprop the detection module
  local grad_det_out = {grad_roi_features, self.grad_roi_boxes}
  local grad_det_in
  if region_proposals then
    -- backward the FixedDetectionModule
    grad_det_in = self.nets.fixed_detection_module:backward(
                      {cnn_features, region_proposals, data.target_boxes, data.target_seqs},
                      grad_det_out)
  else
    -- backward the DetectionModule
    grad_det_in = self.nets.detection_module:backward(
                            {cnn_features, data.target_boxes, data.target_seqs},
                            grad_det_out)
  end
  local grad_cnn_features = grad_det_in[1] -- the [2],[3] are dummy grads
  
  if data.cnn_features then
    data.grad_cnn_features = grad_cnn_features
  end

  -- Maybe run CNN backward (and only the second part of the ConvNet)
  if self.cnn_backward then
    local grad_conv_net1_top
    self:timeit('cnn:backward', function()
      grad_conv_net1_top = self.nets.conv_net2:backward(cnn_features1, grad_cnn_features)
    end)
  end
  
  -- Copy over times from detection module
  if self.timing then
    for k, v in pairs(self.nets.detection_module.stats.times) do
      self.stats.times[k] = v
    end
  end

  -- Maybe dump vars
  if self.dump_vars then
    self.stats.vars.images = data.images
    self.stats.vars.target_boxes = data.target_boxes[1]
    self.stats.vars.target_cls = data.target_seqs[1]
    for k, v in pairs(self.nets.detection_module.stats.vars) do
      self.stats.vars[k] = v
    end
  end

  -- Compute the final full loss
  local total_loss = 0
  self.stats.losses.total_loss = 0
  for k, v in pairs(self.stats.losses) do
    total_loss = total_loss + v
  end
  self.stats.losses.total_loss = total_loss
  -- We're done, weeeee!
  return self.stats.losses, self.stats
end


function DenseCapModel:getParameters(arg)
  -- note, getParameters instead of parameters() is implemented since StnDetectionModel is not a Module
  arg.with_cnn = utils.getopt(arg, 'with_cnn', false)
  arg.with_rpn = utils.getopt(arg, 'with_rpn', true)
  arg.with_recog = utils.getopt(arg, 'with_recog', true)
  local fakenet = nn.Sequential()
  if arg.with_cnn then fakenet:add(self.nets.conv_net2) end -- note: we only return conv_net2, not conv_net1
  if arg.with_rpn then fakenet:add(self.nets.detection_module) end
  if arg.with_recog then 
    fakenet:add(self.nets.recog_base)
    fakenet:add(self.nets.recog_class)
    fakenet:add(self.nets.recog_box)
    fakenet:add(self.nets.recog_lm_encode)
    fakenet:add(self.nets.lm_model)
  end
  return fakenet:getParameters()
end

function DenseCapModel:getParametersSeparate(arg)
  -- returns CNN parameters separately
  arg.with_cnn = utils.getopt(arg, 'with_cnn', false)
  arg.with_rpn = utils.getopt(arg, 'with_rpn', true)
  arg.with_recog = utils.getopt(arg, 'with_recog', true)
  
  -- serialize convnet params maybe
  local cnn_params, cnn_grad_params
  if arg.with_cnn then
    cnn_params, cnn_grad_params = self.nets.conv_net2:getParameters()
  end

  -- serialize rest params
  local fakenet = nn.Sequential()
  if arg.with_rpn then fakenet:add(self.nets.detection_module) end
  if arg.with_recog then 
    fakenet:add(self.nets.recog_base)
    fakenet:add(self.nets.recog_class)
    fakenet:add(self.nets.recog_box)
    fakenet:add(self.nets.recog_lm_encode)
    fakenet:add(self.nets.lm_model)
  end
  local params, grad_params = fakenet:getParameters()

  -- return both separately
  return params, grad_params, cnn_params, cnn_grad_params
end


--[[
Run a test-time forward pass of the model to compute outputs from an image.

Inputs:
- img: 1 x 3 x H x W tensor of pixel data
- arg: Table of arguments
- region_proposals: 1 x B x 5 array of region proposals in (xc, yc, w, h, score) format.
  If these are provided then they will be used; if not then the RPN will be used
  to generate our own region proposals.
--]]
function DenseCapModel:forward_test(img, arg, region_proposals)
  self:evaluate()
  arg = arg or {}
  arg.clip_boxes = utils.getopt(arg, 'clip_boxes', true)
  arg.nms_thresh = utils.getopt(arg, 'nms_thresh', 0.7)
  arg.max_proposals = utils.getopt(arg, 'max_proposals', 300)
  arg.final_nms_thresh = utils.getopt(arg, 'final_nms_thresh', 0.3)
  arg.detections_per_image = utils.getopt(arg, 'detections_per_image', -1)
  arg.clip_final_boxes = utils.getopt(arg, 'clip_final_boxes', 1)
  arg.run_recognition_net = utils.getopt(arg, 'run_recognition_net', 1)
  if arg.detections_per_image == -1 then arg.detections_per_image = nil end

  -- Ensure that the size is (1, 3, H, W).
  assert(img:dim() == 3 or img:dim() == 4)
  if img:dim() == 3 then
    img = img:view(1, img:size(1), img:size(2), img:size(3))
  end
  assert(img:size(1) == 1)
  assert(img:size(2) == 3)

  -- Cast image to correct type
  img = img:type(self.opt.dtype)
  -- Run cnn forward
  local cnn_features1, cnn_features
  self:timeit('test:cnn_forward', function()
    cnn_features1 = self.nets.conv_net1:forward(img)
    cnn_features = self.nets.conv_net2:forward(cnn_features1)
  end)
      
  -- Run detection module forward in test mode to get ROI features
  local image_height, image_width = img:size(3), img:size(4)
  local roi_features, rpn_boxes, obj_scores
  if region_proposals then
    -- Got external region proposals, so get features for them using FixedDetectionModule
    self.nets.fixed_detection_module:setImageSize(image_height, image_width)
    self.nets.fixed_detection_module.dump_vars = self.dump_vars
    roi_features, rpn_boxes = self.nets.fixed_detection_module:forward_test(cnn_features, region_proposals, arg)
    obj_scores = region_proposals[{ 1, {}, 5 }]:contiguous()
  else
    -- Did not get external region proposals, so compute our own using DetectionModule
    self.nets.detection_module.timing = self.timing
    self.nets.detection_module:setImageSize(image_height, image_width)
    self.nets.detection_module.dump_vars = self.dump_vars

    self:timeit('test:det_mod_forward', function()
      roi_features, rpn_boxes, obj_scores = self.nets.detection_module:forward_test(cnn_features, arg)
    end)
    
    for k, v in pairs(self.nets.detection_module.stats.times) do
      self.stats.times[k] = v
    end
  end
  local B3 = rpn_boxes:size(1) -- rpn_boxes is (B3, 4)

  -- This is SUPER GROSS, but we sometimes want to access the region proposals from outside this class,
  -- for example if we are using a trained RPN model to extract region proposals to train a Fast R-CNN model.
  -- TODO: refactor this to not be horrible.
  self.rpn_boxes = rpn_boxes
  self.obj_scores = obj_scores

  -- Forward the recognition base network; cache the codes to use elsewhere
  local roi_codes
  self:timeit('test:recog_base_forward', function()
    roi_codes = self.nets.recog_base:forward(roi_features) -- e.g. returns (B3, 4096) matrix output from VGG FC layers
  end)
  self.roi_codes = roi_codes

  if arg.run_recognition_net == 0 then
    -- return early
    return nil
  end

  -- forward the class scores
  local class_scores = self.nets.recog_class:forward(roi_codes) -- e.g. (B3,C) or (B3,2)
  -- forward the box offsets
  local class_trans = self.nets.recog_box:forward(roi_codes) -- e.g. (B3, C, 4) in detection or (B3, 4) in captioning
  local final_boxes

  final_boxes = self.nets.apply_box_transform:forward{rpn_boxes, class_trans}

  -- Maybe do NMS
  if arg.final_nms_thresh > 0 then
    local final_boxes_float = final_boxes:float()
    local class_scores_float = class_scores:float()
    local boxes_scores = torch.FloatTensor(final_boxes:size(1), 5)
    local boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(final_boxes_float)
    boxes_scores[{{}, {1, 4}}]:copy(boxes_x1y1x2y2)
    boxes_scores[{{}, 5}]:copy(class_scores_float[{{}, 1}])
    local idx
    self:timeit('test:final_nms', function()
        idx = box_utils.nms(boxes_scores, arg.final_nms_thresh, arg.detections_per_image)
      end)
    -- Since index is really slow for CudaTensors we just do it on CPU then copy back to gpu
    final_boxes = final_boxes_float:index(1, idx):typeAs(final_boxes)
    class_scores = class_scores_float:index(1, idx):typeAs(class_scores)
    -- NOTE: we are keeping self.obj_scores in before-nms size, but cropping the local obj_scores
    -- variable to be returned in visualization
    obj_scores = obj_scores:index(1, idx) 
    roi_codes = roi_codes:index(1, idx)
    self.roi_codes = roi_codes -- gross. ew.

    -- We also need to apply the final nms to the vars that DetectionModel dumped
    -- so that we can properly visualize the anchors and rpns for our final detections.
    -- This is kinda gross because it makes StnDetectionModule and DetectionModule even
    -- more coupled, but that's too bad.
    local det_mod_vars = self.nets.detection_module.stats.vars
    local var_names = {'test_rpn_boxes_nms', 'test_rpn_anchors_nms', 'test_rpn_scores_nms'}
    for _, var_name in ipairs(var_names) do
      if det_mod_vars[var_name] then
        det_mod_vars[var_name] = det_mod_vars[var_name]:index(1, idx)
      end
    end
  end

  -- forward the sequence RNN in captioning
  local seq
  -- we have to be careful and use batches here or the RNN will blow up memory
  self:timeit('test:rnn_forward', function()
    local batch_size = 100
    local n = roi_codes:size(1)
    seq = torch.LongTensor(self.opt.seq_length, n)
    for i=1,n,batch_size do
      local batch_range = {i,i+100-1}
      if batch_range[2] > n then batch_range[2] = n end -- clamp in case batch_size doesn't divide
      local roi_codes_encoded = self.nets.recog_lm_encode:forward(roi_codes[{batch_range}])
      local seq_batch = self.nets.lm_model:sample(roi_codes_encoded)
      seq[{ {}, batch_range }] = seq_batch
    end
  end)

  -- Maybe clip final boxes to image boundary
  if arg.clip_final_boxes == 1 then
    local image_bounds = {
      x_min=1, x_max=img:size(4),
      y_min=1, y_max=img:size(3)
    }
    local oob_mask
    final_boxes, oob_mask = box_utils.clip_boxes(final_boxes, image_bounds, 'xcycwh')
    -- For now we'll just assert that all final boxes need to be in bounds;
    -- if this actually is not true in practice then we can use the oob_mask to pick out
    -- elements from final_boxes, class_scores, and seq.
    if oob_mask:sum() ~= oob_mask:nElement() then
      local valid_idx = oob_mask:nonzero()
    end
    if oob_mask:sum() ~= oob_mask:nElement() then
      print('WARNING: SOME FINAL BOXES WERE OUT OF BOUNDS')
    end
    --assert(oob_mask:sum() == oob_mask:nElement(), 'Some final boxes were out of bounds')
  end

  if self.dump_vars then
    if region_proposals then
      for k, v in pairs(self.nets.fixed_detection_module.stats.vars) do
        self.stats.vars[k] = v
      end
    else
      -- copy over all vars from detection_module
      for k, v in pairs(self.nets.detection_module.stats.vars) do
        self.stats.vars[k] = v
      end
    end
    -- add some of our own
    self.stats.vars.test_final_boxes = final_boxes
    self.stats.vars.class_scores = class_scores
    self.stats.vars.obj_scores = obj_scores
    if seq then
      self.stats.vars.seq = seq -- also encode seq, if appropriate
    end
  end

  -- final boxes is (B3, C, 4) or (B3, 4), class_scores is (B3, C), 
  -- and seq is (K, B3) where K is self.opt.seq_length (or nil in detection)
  return final_boxes, class_scores, seq
end


function DenseCapModel:get_region_features(img, arg)
  arg = arg or {}
  arg.num_regions = utils.getopt(arg, 'num_regions', 300)
  arg.region_nms_thresh = utils.getopt(arg, 'region_nms_thresh', 0.7)

  local forward_test_arg = {
      max_proposals = arg.num_regions,
      final_nms_thresh = arg.region_nms_thresh,
  }
  local final_boxes, class_scores, seq = self:forward_test(img, forward_test_arg)
  local roi_codes = self.roi_codes
  local B3 = final_boxes:size(1)

  -- Now final_boxes is (B3, 4), class_scores is (B3, C).
  local boxes_scores = final_boxes.new(B3, 5)
  local boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(final_boxes)
  boxes_scores[{{}, {1, 4}}]:copy(boxes_x1y1x2y2)

  -- First class is positives
  boxes_scores[{{}, 5}]:copy(class_scores[{{}, 1}])

  local idx = box_utils.nms(boxes_scores, arg.region_nms_thresh, arg.num_regions)
  return final_boxes:index(1, idx), roi_codes:index(1, idx)
end

------------------------------------------------------------------------------------------
--[[
Given an image and some labels, find the argmax grounding of those labels into the image.

Inputs:
img: Tensor of shape (1, 3, H, W) giving pixel data for the image.
labels: Either a tensor of shape (1, B, L) giving labels, or a list of k such Tensors.

Returns:
- boxes: A Tensor of shape (B, 4) giving the aligned box for each label, or a list of
  k such Tensors if labels was a list.
- scores: A number giving the match score between the image and the labels, or a Tensor
  of shape k giving match scores for each element of labels if it was a list.
  Scores are negative log-probabilities, so SMALLER indicates a BETTER match
--]]
function DenseCapModel:ground_labels(img, labels)
  local tensor_input = false
  if torch.isTensor(labels) then
    labels = {labels}
    tensor_input = true
  end

  local boxes, roi_codes = self:get_region_features(img)
  local roi_codes_encoded = self.nets.recog_lm_encode:forward(roi_codes)

  local all_aligned_boxes = {}
  local all_match_scores = torch.Tensor(#labels)

  local num_boxes = boxes:size(1)
  
  for j = 1, #labels do
    print(string.format('Starting query %d / %d', j, #labels))
    local this_label = labels[j]
    local num_labels = this_label:size(2)
    local seq_length = this_label:size(3)
    local box_label_logprobs = torch.zeros(num_boxes, num_labels)

    -- best_idxs[i] = j means that labels[i] alignes to boxes[j]
    -- match_scores[i] = s means that label[i] aligning to box[j] has a score of s
    local best_idxs = torch.LongTensor(num_labels)
    local match_scores = torch.DoubleTensor(num_labels)

    for i = 1, this_label:size(2) do
      local labels_expand = this_label[{1, i}]:view(-1, 1):expand(seq_length, num_boxes)
      -- print(string.format('i = %d, j = %d, calling lm_model forward', i, j))
      local lm_scores = self.nets.lm_model:forward{roi_codes_encoded, labels_expand}
      -- process the criterion and the loss
      local crit = self.nets.lm_crit
      local loss = crit:forward(lm_scores, labels_expand)
      local score, idx = torch.min(crit.losses, 1)
      best_idxs[i] = idx[1]
      match_scores[i] = score[1]
    end

    -- The call to index allocates a new tensor so we don't have to worry about the
    -- next forward pass running over our memory.
    table.insert(all_aligned_boxes, boxes:index(1, best_idxs))
    all_match_scores[j] = match_scores:mean()
  end

  if tensor_input then
    return all_aligned_boxes[1], all_match_scores[1]
  else
    return all_aligned_boxes, all_match_scores
  end
end


--[[
Given an image, some boxes, and a label for each box, return a compatibility score between
the two (for structured retrieval).

Inputs:
img: Tensor of shape (1, 3, H, W) giving pixel data for the image.
boxes: Tensor of shape (B, 4) giving coordinates of boxes in (xc, yc, w, h) format
labels: Tensor of shape (B, L) giving desired label for each box; for detection
        L=1 and each label is an object class; for dense captioning L>1 and each
        label is a caption.
--]]
function DenseCapModel:get_label_score(img, boxes, labels)
  assert(img:dim() == 4)
  assert(img:size(1) == 1 and img:size(2) == 3)

  -- Cast inputs to proper datatype
  img = img:type(self.opt.dtype)
  boxes = boxes:type(self.opt.dtype)

  -- Forward the CNN to get features for the whole image
  local cnn_features1 = self.nets.conv_net1:forward(img)
  local cnn_features = self.nets.conv_net2:forward(cnn_features1)

  -- We need an roi pooling layer to pick out features for the boxes
  -- TODO should we cache this in self?
  local roi_pooling = nn.BilinearRoiPooling(self.opt.output_height, self.opt.output_width)
  roi_pooling:type(self.opt.dtype)

  -- Forward the RoI pooling layer to get codes for the specified boxes
  roi_pooling:setImageSize(img:size(3), img:size(4))
  local roi_features = roi_pooling:forward{cnn_features[1], boxes}

  -- Forward the recog base network to get roi codes
  local roi_codes = self.nets.recog_base:forward(roi_features)

  local loss = 0
  local roi_codes_encoded = self.nets.recog_lm_encode:forward(roi_codes)
  -- Language model wants labels transposed for efficiency
  local seq_transposed = labels:transpose(1,2):contiguous()
  local lm_scores = self.nets.lm_model:forward{roi_codes_encoded, seq_transposed}
  loss = self.nets.lm_crit:forward(lm_scores, seq_transposed)

  return -loss
end

