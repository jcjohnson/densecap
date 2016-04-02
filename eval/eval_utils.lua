
local cjson = require 'cjson'
local utils = require 'densecap.utils'

local eval_utils = {}

function eval_utils.score_captions(records)
  --[[
  function takes a table of records, in form:
    {'key': {'references':{'',..,''}, 'candidate':''}, ...}
  and returns a blob table result in form:
    {'scores': {'key' : score}, 'average_score': float}
  --]]

  -- serialize records to json file
  utils.write_json('eval/input.json', records)
  -- invoke python process 
  os.execute('python eval/meteor_bridge.py')
  -- read out results
  local blob = utils.read_json('eval/output.json')

  return blob
end

return eval_utils
