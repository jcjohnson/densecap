require 'torch'
require 'nn'
require 'nngraph'

require 'densecap.LanguageModel'
require 'densecap.LocalizationLayer'
require 'densecap.modules.BoxRegressionCriterion'
require 'densecap.modules.BilinearRoiPooling'
require 'densecap.modules.ApplyBoxTransform'
require 'densecap.modules.LogisticCriterion'
require 'densecap.modules.PosSlicer'

local box_utils = require 'densecap.box_utils'
local net_utils = require 'densecap.net_utils'
local utils = require 'densecap.utils'


local DenseCapModel, parent = torch.class('nn.DenseCapModel', 'nn.Module')


function DenseCapModel:__init(opt)
  opt = opt or {}  
  opt.cnn_name = utils.getopt(opt, 'cnn_name', 'vgg-16')
  opt.backend = utils.getopt(opt, 'backend', 'cudnn')
  opt.path_offset = utils.getopt(opt, 'path_offset', '')
  opt.dtype = utils.getopt(opt, 'dtype', 'torch.CudaTensor')
  opt.vocab_size = utils.getopt(opt, 'vocab_size')
  opt.std = utils.getopt(opt, 'std', 0.01) -- Used to initialize new layers

  -- For test-time handling of final boxes
  opt.final_box_nms = utils.getopt(opt, 'final_nms_thresh', 0.3)

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
  self.opt = opt -- TODO: this is... naughty. Do we want to create a copy instead?
  
  -- This will hold various components of the model
  self.nets = {}
  
  -- This will hold the whole model
  self.net = nn.Sequential()
  
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
  self.net:add(self.nets.conv_net1)
  self.net:add(self.nets.conv_net2)
  
  -- Figure out the receptive fields of the CNN
  -- TODO: Should we just hardcode this too per CNN?
  local conv_full = net_utils.subsequence(cnn, conv_start1, conv_end2)
  local x0, y0, sx, sy = net_utils.compute_field_centers(conv_full)
  self.opt.field_centers = {x0, y0, sx, sy}

  self.nets.localization_layer = nn.LocalizationLayer(opt)
  self.net:add(self.nets.localization_layer)
  
  -- Recognition base network; FC layers from VGG.
  -- Produces roi_codes of dimension fc_dim.
  -- TODO: Initialize this from scratch for ResNet?
  self.nets.recog_base = net_utils.subsequence(cnn, recog_start, recog_end)
  
  -- Objectness branch; outputs positive / negative probabilities for final boxes
  self.nets.objectness_branch = nn.Linear(fc_dim, 2)
  self.nets.objectness_branch.weight:normal(0, opt.std)
  self.nets.objectness_branch.bias:zero()
  
  -- Final box regression branch; regresses from RPN boxes to final boxes
  self.nets.box_reg_branch = nn.Linear(fc_dim, 4)
  self.nets.box_reg_branch.weight:zero()
  self.nets.box_reg_branch.bias:zero()

  -- Set up LanguageModel
  local lm_opt = {
    vocab_size = opt.vocab_size,
    input_encoding_size = opt.rnn_encoding_size,
    rnn_size = opt.rnn_size,
    seq_length = opt.seq_length,
    idx_to_token = opt.idx_to_token,
    image_vector_dim=fc_dim,
  }
  self.nets.language_model = nn.LanguageModel(lm_opt)

  self.nets.recog_net = self:_buildRecognitionNet()
  self.net:add(self.nets.recog_net)

  -- Set up Criterions
  self.crits = {}
  self.crits.objectness_crit = nn.LogisticCriterion()
  self.crits.box_reg_crit = nn.BoxRegressionCriterion(opt.end_box_reg_weight)
  self.lm_crit = nn.TemporalCrossEntropyCriterion()

  self:training()
end


function DenseCapModel:_buildRecognitionNet()
  local roi_feats = nn.Identity()()
  local roi_boxes = nn.Identity()()
  local gt_seq = nn.Identity()()
  local gt_boxes = nn.Identity()()

  local roi_codes = self.nets.recog_base(roi_feats)
  local objectness_scores = self.nets.objectness_branch(roi_codes)

  local pos_roi_codes = nn.PosSlicer(){roi_codes, gt_seq}
  local pos_roi_boxes = nn.PosSlicer(){roi_boxes, gt_boxes}
  
  local final_box_trans = self.nets.box_reg_branch(pos_roi_codes)
  local final_boxes = nn.ApplyBoxTransform(){pos_roi_boxes, final_box_trans}

  local lm_input = {pos_roi_codes, gt_seq}
  local lm_output = self.nets.language_model(lm_input)

  local inputs = {roi_feats, roi_boxes, gt_seq, gt_boxes}
  local outputs = {
    objectness_scores,
    pos_roi_boxes, final_box_trans, final_boxes,
    lm_output,
    gt_boxes, gt_seq,
  }
  return nn.gModule(inputs, outputs)
end


function DenseCapModel:training()
  parent.training(self)
  self.net:training()
end


function DenseCapModel:evaluate()
  parent.evaluate(self)
  self.net:evaluate()
end


function DenseCapModel:updateOutput(input)
  -- Make sure the input is (1, 3, H, W)
  assert(input:dim() == 4 and input:size(1) == 1 and input:size(2) == 3)
  local H, W = input:size(3), input:size(4)
  self.nets.localization_layer:setImageSize(H, W)

  if self.train then
    assert(not self._called_forward,
      'Must call setGroundTruth before training-time forward pass')
    self._called_forward = true
  end
  self.output = self.net:forward(input)

  -- At test-time, apply NMS to final boxes
  if opt.final_nms_thresh > 0 then
    local final_boxes_float = self.output[4]:float()
    local class_scores_float = self.output[1]:float()
    local boxes_scores = torch.FloatTensor(final_boxes:size(1), 5)
    local boxes_x1y1x2y2 = box_utils.xcycwh_to_x1y1x2y2(final_boxes_float)
    boxes_scores[{{}, {1, 4}}]:copy(boxes_x1y1x2y2)
    boxes_scores[{{}, 5}]:copy(class_scores_float[{{}, 1}])
    local idx = box_utils.nms(boxes_scores, self.opt.final_nms_thresh)
    self.output[4] = final_boxes_float:index(1, idx):typeAs(self.output[4])
    self.output[1] = class_scores_float:index(1, idx):typeAs(self.output[1])

    -- TODO: In the old StnDetectionModel we also applied NMS to the
    -- variables dumped by the LocalizationLayer. Do we want to do that?
  end
end


function DenseCapModel:forward_train(input)
  assert(not self._called_forward,
    'Must call setGroundTruth before training-time forward pass')
  self._called_forward = true

  self.output = self.net:forward(input)
  return self.output
end


function DenseCapModel:setGroundTruth(gt_boxes, gt_labels)
  self.gt_boxes = gt_boxes
  self.gt_labels = gt_labels
  self._called_forward = false
  self.nets.localization_layer:setGroundTruth(gt_boxes, gt_labels)
end

