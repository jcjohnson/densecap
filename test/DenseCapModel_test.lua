require 'densecap.DenseCapModel'


local tests = torch.TestSuite()
local tester = torch.Tester()


function tests.simpleTest()
  local dtype = 'torch.CudaTensor'
  
  if dtype == 'torch.CudaTensor' then
    require 'cutorch'
    require 'cunn'
  end
  local L, V = 10, 100
  local opt = {
    vocab_size=V,
    mid_box_reg_weight=0.1,
    mid_objectness_weight=0.1,
    end_box_reg_weight=1.0,
    end_objectness_weight=1.0,
    captioning_weight=1.0,
    idx_to_token = {},
    seq_length=L,
    rnn_encoding_size=64,
  }
  local model = nn.DenseCapModel(opt):type(dtype)

  local H, W, B = 480, 640, 45
  local img = torch.randn(1, 3, H, W):type(dtype)
  local gt_boxes = torch.randn(1, B, 4):add(1.0):mul(100):abs():type(dtype)
  local gt_labels = torch.LongTensor(1, B, L):random(V):type(dtype)

  model:forward_backward{
    image=img,
    gt_boxes=gt_boxes,
    gt_labels=gt_labels,
  }
  --[[
  -- Training time forward pass
  model:setGroundTruth(gt_boxes, gt_labels)
  local out = model:forward(img)
  print(out)

  -- Test-time forward pass
  model:evaluate()
  local out = model:forward(img)
  print(out)

  print(model.net)
  --]]
end


tester:add(tests)
tester:run()

