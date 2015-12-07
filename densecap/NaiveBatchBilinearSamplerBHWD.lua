require 'nn'
require 'stn'

local layer, parent = torch.class('nn.NaiveBatchBilinearSamplerBHWD',
                                  'nn.Module')

--[[
  NaiveBatchBilinearSamplerBHWD performs bilinear sampling to pull out
  multiple patches from a single input image.

  Inputs:
  - inputImages: Tensor of shape (H, W, C)
  - grids: Tensor of shape (N, HH, WW, 2)

  Output:
  - Tensor of shape (N, HH, WW, C) which is the result of applying each
    sampling grid to the input image.

  This implementation is very naive and inefficient, and is mainly used
  to test the correctness of the more efficient implementation of the same
  function in BatchBilinearSamplerBHWD.lua.

  For more details see the discussion in BatchBilinearSamplerBHWD.lua.
--]]

function layer:__init()
  parent.__init(self)
  self.sampler = nn.BilinearSamplerBHWD()
  self.replicate = nil
  self.replicate_out = nil
end

function layer:updateOutput(input)
  local feats, grids = input[1], input[2]
  local B = grids:size(1)
  self.replicate = nn.Replicate(grids:size(1)):type(feats:type())
  self.replicate_out = self.replicate:forward(feats)
  self.output = self.sampler:forward{self.replicate_out, grids}
  return self.output
end

function layer:updateGradInput(input, gradOutput)
  local feats, grids = input[1], input[2]
  local grad_sampler_in = self.sampler:backward(
                                {self.replicate_out, grids},
                                gradOutput)
  local grad_feats = self.replicate:backward(feats, grad_sampler_in[1])
  self.gradInput = {grad_feats, grad_sampler_in[2]}
  return self.gradInput
end
