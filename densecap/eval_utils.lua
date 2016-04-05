--[[
Utilities and function for evaluating the Dense Captioning metric, 
which combines both object detection and captioning performance into one number.

The sligthly gross part is that we want to use METEOR in our evaluation.
--]]

local box_utils = require 'densecap.box_utils'
local utils = require 'densecap.utils'
local eval_utils = {}

local function python_eval(records, all_target_text, all_predict_text, id)
  
  local out = {}
  out['records'] = records
  out['all_target_text'] = all_target_text
  out['all_predict_text'] = all_predict_text

  -- dump json to file
  utils.write_json('coco_eval/coco-caption/blob' .. id .. '.json', out)
  -- now call python process which will open the blob and insert all the scores in place
  os.execute('./call_python_caption_eval.sh ' .. id)
  -- read back the result
  local result_struct = utils.read_json('coco_eval/coco-caption/blob' .. id .. '_out.json')
  local annotated_records = result_struct['annotated_records']
  local lang_score = result_struct['lang_score']

  return annotated_records, lang_score
end

local DenseCaptionEvaluator = torch.class('DenseCaptionEvaluator')
function DenseCaptionEvaluator:__init(opt)
  self.all_logprobs = {}
  self.records = {}
  self.n = 1
  self.npos = 0

  self.all_target_text = {}
  self.all_predict_text = {}

  self.id = opt.id
end

function pluck_boxes(ix, boxes, text)
  -- ix is a list (length N) of LongTensors giving indices to boxes/text. Use them to do merge
  -- return boxes Nx4, and text[] of length N

  local N = #ix
  local new_boxes = torch.zeros(N, 4)
  local new_text = {}

  for i=1,N do
    
    local ixi = ix[i]
    local n = ixi:nElement()
    local bsub = boxes:index(1, ixi)
    local newbox = torch.mean(bsub, 1)
    new_boxes[i] = newbox

    local texts = {}
    if text then
      for j=1,n do
        table.insert(texts, text[ixi[j]])
      end
    end
    table.insert(new_text, texts)
  end

  return new_boxes, new_text
end

-- boxes is (B x 4) are xcycwh, logprobs are (B x C)
-- target_boxes (Mx4), and target_cls (Mx1)
-- boxes is an array (possibly on GPU) of size Nx4, in xcycwh format
-- to compute the result in torch/lua, without having ot go to Matlab. This translation
-- is kept as close as possible to the original.
function DenseCaptionEvaluator:addResult(info, boxes, logprobs, target_boxes, target_text, predict_text)

  boxes = box_utils.xcycwh_to_x1y1x2y2(boxes) -- voc format is x1,y1,x2,y2
  target_boxes = box_utils.xcycwh_to_x1y1x2y2(target_boxes) -- x1,y1,x2,y2

  -- make sure we're on CPU
  boxes = boxes:float()
  logprobs = logprobs[{ {}, 1 }]:double() -- grab the positives class (1)
  target_boxes = target_boxes:float()

  -- merge ground truth boxes that overlap by >= 0.7
  local mergeix = box_utils.merge_boxes(target_boxes, 0.7) -- merge groups of boxes together
  local merged_boxes, merged_text = pluck_boxes(mergeix, target_boxes, target_text)
  --local merged_boxes = target_boxes
  --local merged_text = target_text

  -- 1. Sort detections by decreasing confidence
  local Y,IX = torch.sort(logprobs,1,true) -- true makes order descending
  local nd = logprobs:size(1) -- number of detections
  local nt = merged_boxes:size(1) -- number of gt boxes
  assert(boxes:nDimension() == 2)
  
  local used = torch.zeros(nt)
  for d=1,nd do -- for each detection in descending order of confidence
    local ii = IX[d]
    local bb = boxes[ii]
    
    -- assign the box to its best match in true boxes
    local ovmax = 0
    local jmax = -1
    for j=1,nt do
      local bbgt = merged_boxes[j]
      local bi = {math.max(bb[1],bbgt[1]), math.max(bb[2],bbgt[2]),
                  math.min(bb[3],bbgt[3]), math.min(bb[4],bbgt[4])}
      local iw = bi[3]-bi[1]+1
      local ih = bi[4]-bi[2]+1
      if iw>0 and ih>0 then
        -- compute overlap as area of intersection / area of union
        local ua = (bb[3]-bb[1]+1)*(bb[4]-bb[2]+1)+
                   (bbgt[3]-bbgt[1]+1)*(bbgt[4]-bbgt[2]+1)-iw*ih
        local ov = iw*ih/ua
        if ov > ovmax then
          ovmax = ov
          jmax = j
        end
      end
    end

    local ok = 1
    if used[jmax] == 0 then
      used[jmax] = 1 -- mark as taken
    else
      ok = 0
    end

    -- record the best box, the overlap, and the fact that we need to score the language match
    local record = {}
    record.ok = ok
    record.ov = ovmax
    record.pred = predict_text[ii]
    record.gt = merged_text[jmax] -- will be nil if jmax stays -1
    record.imgid = self.n
    table.insert(self.records, record)
  end
  
  table.insert(self.all_target_text, target_text) -- have to keep track of these as well
  table.insert(self.all_predict_text, predict_text)

  -- keep track of results
  self.n = self.n + 1
  self.npos = self.npos + nt
  table.insert(self.all_logprobs, Y:double()) -- inserting the sorted logprobs as double
end

function DenseCaptionEvaluator:numAdded()
  return self.n - 1
end

function DenseCaptionEvaluator:evaluate(verbose)
  if verbose == nil then verbose = true end

  local min_overlaps = {0.3, 0.4, 0.5, 0.6, 0.7}
  local min_scores = {-1, 0, 0.05, 0.1, 0.15, 0.2, 0.25} -- note: -1 will ALWAYS match since scores >= 0, can be used to eval detection alone
  local score_type = 'METEOR'

  -- concatenate everything across all images
  local logprobs = torch.cat(self.all_logprobs, 1) -- concat all logprobs
  -- call python to evaluate all records and get their BLEU/METEOR scores
  local lang_score
  self.records, lang_score = python_eval(self.records, self.all_target_text, self.all_predict_text, self.id) -- replace in place (prev struct will be collected)
  collectgarbage()
  collectgarbage()
  
  -- prints/debugging
  if verbose then
    for k=1,#self.records do
      local record = self.records[k]
      if record.ov > 0 and record.ok == 1 and k % 1000 == 0 then
        local txtgt = ''
        if type(record.gt) == "table" then
          for kk,vv in pairs(record.gt) do txtgt = txtgt .. vv .. '. ' end
        else
          txtgt = record.gt
        end
        print(string.format('IMG %d PRED: %s, GT: %s, OK: %d, OV: %f SCORE: %f', record.imgid, record.pred, txtgt, record.ok, record.ov, record.scores[score_type]))
      end  
    end
  end

  -- lets now do the evaluation
  local y,ix = torch.sort(logprobs,1,true) -- true makes order descending

  local ap_results = {}
  local det_results = {}
  for foo, min_overlap in pairs(min_overlaps) do
    for foo2, min_score in pairs(min_scores) do

      -- go down the list and build tp,fp arrays
      local n = y:nElement()
      local tp = torch.zeros(n)
      local fp = torch.zeros(n)
      for i=1,n do
        -- pull up the relevant record
        local ii = ix[i]
        local r = self.records[ii]

        if not r.gt then 
          fp[i] = 1 -- nothing aligned to this predicted box in the ground truth
        else
          -- ok something aligned. Lets check if it aligned enough, and correctly enough
          local score = r.scores[score_type]
          if r.ov >= min_overlap and r.ok == 1 and score > min_score then
            tp[i] = 1
          else
            fp[i] = 1
          end
        end
      end

      fp = torch.cumsum(fp,1)
      tp = torch.cumsum(tp,1)
      local rec = torch.div(tp, self.npos)
      local prec = torch.cdiv(tp, fp + tp)

      -- compute max-interpolated average precision
      local ap = 0
      local apn = 0
      for t=0,1,0.01 do
        local mask = torch.ge(rec, t):double()
        local prec_masked = torch.cmul(prec, mask)
        local p = torch.max(prec_masked)
        ap = ap + p
        apn = apn + 1
      end
      ap = ap / apn

      -- store it
      if min_score == -1 then
        det_results['ov' .. min_overlap] = ap
      else
        ap_results['ov' .. min_overlap .. '_score' .. min_score] = ap
      end
    end
  end

  local map = utils.average_values(ap_results)
  local detmap = utils.average_values(det_results)

  -- lets get out of here
  local results = {map = map, ap_breakdown = ap_results, detmap = detmap, det_breakdown = det_results, lang_score = lang_score}
  return results
end

local function table_avg(t)

end

return eval_utils

