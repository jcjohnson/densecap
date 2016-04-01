require 'nn'
require 'torch-rnn'

local utils = require 'densecap.utils'


local LM, parent = torch.class('nn.LanguageModel', 'nn.Module')


function LM:__init(opt)
  parent.__init(self)

  opt = opt or {}
  self.vocab_size = utils.getopt(opt, 'vocab_size')
  self.input_encoding_size = utils.getopt(opt, 'input_encoding_size')
  self.image_vector_dim = utils.getopt(opt, 'image_vector_dim')
  self.rnn_size = utils.getopt(opt, 'rnn_size')
  self.seq_length = utils.getopt(opt, 'seq_length')
  self.num_layers = utils.getopt(opt, 'num_layers', 1)
  self.idx_to_token = utils.getopt(opt, 'idx_to_token')
  self.dropout = utils.getopt(opt, 'dropout', 0)

  -- For mapping from image vectors to word vectors
  self.image_encoder = nn.Sequential()
  self.image_encoder:add(nn.Linear(self.image_vector_dim, self.input_encoding_size))
  self.image_encoder:add(nn.ReLU(true))
  self.image_encoder:add(nn.View(1, -1):setNumInputDims(1))
  
  self.START_TOKEN = self.vocab_size + 1
  self.END_TOKEN = self.vocab_size + 1
  self.NULL_TOKEN = self.vocab_size + 2

  -- For mapping word indices to word vectors
  self.lookup_table = nn.LookupTable(self.vocab_size + 2, self.input_encoding_size)
  
  -- Change this to sample from the distribution instead
  self.sample_argmax = true

  -- self.rnn maps wordvecs of shape N x T x D to word probabilities of shape N x T x V
  self.rnn = nn.Sequential()
  for i = 1, self.num_layers do
    local input_dim = self.rnn_size
    if i == 1 then
      input_dim = self.input_encoding_size
    end
    self.rnn:add(nn.LSTM(input_dim, self.rnn_size))
    if self.dropout > 0 then
      self.rnn:add(nn.Dropout(self.dropout))
    end
  end

  self.view_in = nn.View(1, 1, -1):setNumInputDims(3)
  self.view_out = nn.View(1, -1):setNumInputDims(2)
  self.rnn:add(self.view_in)
  self.rnn:add(nn.Linear(self.rnn_size, self.vocab_size + 1))
  self.rnn:add(self.view_out)

  -- self.net maps a table {image_vecs, gt_seq} to word probabilities
  self.net = nn.Sequential()
  local parallel = nn.ParallelTable()
  parallel:add(self.image_encoder)
  parallel:add(self.start_token_generator)
  parallel:add(self.lookup_table)
  self.net:add(parallel)
  self.net:add(nn.JoinTable(1, 2))
  self.net:add(self.rnn)

  self:training()
end


--[[
Decodes a sequence into a table of strings

Inputs:
- seq: tensor of shape N x T

Returns:
- captions: Array of N strings
--]]
function LM:decodeSequence(seq)
  local delimiter = ' '
  local captions = {}
  local N, T = seq:size(1), seq:size(2)
  for i = 1, N do
    local caption = ''
    for t = 1, T do
      local idx = seq[{i, t}]
      if idx == self.END_TOKEN then break end
      if t > 1 then
        caption = caption .. delimiter
      end
      caption = caption .. self.idx_to_token[idx]
    end
    table.insert(captions, caption)
  end
  return captions
end


function LM:updateOutput(input)
  local image_vectors = input[1]
  local gt_sequence = input[2]
    
  if gt_sequence ~= nil then
    -- Add a start token to the start of the gt_sequence, and replace
    -- 0 with NULL_TOKEN
    local N, T = gt_sequence:size(1), gt_sequence:size(2)
    self._gt_with_start = gt_sequence.new(N, T + 1)
    self._gt_with_start[{{}, 1}]:fill(self.START_TOKEN)
    self._gt_with_start[{{}, {2, T + 1}}]:copy(gt_sequence)
    local mask = torch.eq(self._gt_with_start, 0)
    self._gt_with_start[mask] = self.NULL_TOKEN
    
    -- Reset the views around the nn.Linear
    self.view_in:resetSize(N * (T + 2), -1)
    self.view_out:resetSize(N, T + 2, -1)
    self.output = self.net:updateOutput{image_vectors, self._gt_with_start}
    self._forward_sampled = false
    return self.output
  else
    self._forward_sampled = true
    return self:sample(image_vectors)
  end
end


--[[
Convert a ground-truth sequence of shape to a target suitable for the
TemporalCrossEntropyCriterion from torch-rnn.

Input:
- gt_sequence: Tensor of shape (N, T) where each element is in the range [0, V];
  an entry of 0 is a null token.
--]]
function LM:getTarget(gt_sequence)
  -- Make sure it's on CPU since we will loop over it
  local gt_sequence_long = gt_sequence:long()
  local N, T = gt_sequence:size(1), gt_sequence:size(2)
  local target = torch.LongTensor(N, T + 2):zero()
  target[{{}, {2, T + 1}}]:copy(gt_sequence)
  for i = 1, N do
    for t = 2, T + 2 do
      if target[{i, t}] == 0 then
        -- Replace the first null with an end token
        target[{i, t}] = self.END_TOKEN
        break
      end
    end
  end
  return target:type(gt_sequence:type())
end


function LM:sample(image_vectors)
  local N, T = image_vectors:size(1), self.seq_length
  local seq = torch.LongTensor(N, T):zero()
  local softmax = nn.SoftMax():type(image_vectors:type())
  
  -- During sampling we want our LSTM modules to remember states
  for i = 1, #self.rnn do
    local layer = self.rnn:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = true
    end
  end

  -- Reset view sizes
  self.view_in:resetSize(N, -1)
  self.view_out:resetSize(N, 1, -1)

  -- First timestep: image vectors, ignore output
  local image_vecs_encoded = self.image_encoder:forward(image_vectors)
  self.rnn:forward(image_vecs_encoded)

  -- Now feed words through RNN
  for t = 1, T do
    local words = nil
    if t == 1 then
      -- On the first timestep, feed START tokens
      words = torch.LongTensor(N, 1):fill(self.START_TOKEN)
    else
      -- On subsequent timesteps, feed previously sampled words
      words = seq[{{}, {t-1, t-1}}]
    end
    local wordvecs = self.lookup_table:forward(words)
    local scores = self.rnn:forward(wordvecs):view(N, -1)
    local idx = nil
    if self.sample_argmax then
      _, idx = torch.max(scores, 2)
    else
      local probs = softmax:forward(scores)
      idx = torch.multinomial(probs, 1):view(-1):long()
    end
    seq[{{}, t}]:copy(idx)
  end

  -- After sampling stop remembering states
  for i = 1, #self.rnn do
    local layer = self.rnn:get(i)
    if torch.isTypeOf(layer, nn.LSTM) then
      layer:resetStates()
      layer.remember_states = false
    end
  end

  self.output = seq
  return self.output
end


function LM:backward(input, gradOutput, scale)
  assert(self._forward_sampled == false, 'cannot backprop through sampling')
  input = {input[1], self._gt_with_start}
  self.gradInput = self.net:backward(input, gradOutput, scale)
end


function LM:parameters()
  return self.net:parameters()
end


function LM:training()
  parent.training(self)
  self.net:training()
end


function LM:evaluate()
  parent.evaluate(self)
  self.net:evaluate()
end


function LM:clearState()
  self.net:clearState()
end
