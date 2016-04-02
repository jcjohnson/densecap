
local eval_utils = require 'eval.eval_utils'
local utils = require 'densecap.utils'

local tests = {}
local tester = torch.Tester()

function tests.sanityCheckTest()

  local records = {}
  records['id1'] = {references={'an example ref', 'another ref', 'and one more'}, candidate='one words matches'}
  records['id2'] = {references={'some sentence', 'one more for fun'}, candidate='nothing matches'}
  records['id3'] = {references={'expecting perfect match', 'garbage sent', 'bleh one more'}, candidate='expecting perfect match'}

  local blob = eval_utils.score_captions(records)

  local scores = blob.scores
  tester:asserteq(utils.count_keys(scores), 3)
  tester:assertgt(scores.id1, 0.0)
  tester:assertlt(scores.id1, 1.0)
  tester:asserteq(scores.id2, 0, 'nothing should match')
  tester:asserteq(scores.id3, 1.0, 'exact match expected')
  tester:assertgt(blob.average_score, 0.0, 'average score between 0 and 1')
  tester:assertlt(blob.average_score, 1.0, 'average score between 0 and 1')
end

tester:add(tests)
tester:run()
