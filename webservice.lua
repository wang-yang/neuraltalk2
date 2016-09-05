----------------------------------------------------------------------------------
---- Basic web service using turbo, to caption images using a pre-trained network
----
------------------------------------------------------------------------------------

local turbo = require("turbo")
local torch = require("torch")
local image = require("image")
local nn = require("nn")
local socket = require("socket")
local utils = require('misc.utils')
local image = require('image')
require 'misc.LanguageModel'
local net_utils = require 'misc.net_utils'

-- th eval.lua -gpuid -1 
--             -model model/model_id1-501-1448236541.t7_cpu.t7 
--             -image_folder images/ 
--             -num_images 2

cmd = torch.CmdLine()
cmd:text()
cmd:text('Start image caption service')
cmd:text()
cmd:text('Options')
-- Input paths
cmd:option('-model','','path to model to evaluate')
cmd:option('-batch_size', 1, 'if > 0 then overrule, otherwise load from checkpoint.')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 123, 'random number generator seed to use')
cmd:text()

local opt = cmd:parse(arg)
torch.manualSeed(opt.seed) -- random number generator seed to use
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU
assert(string.len(opt.model) > 0, 'must provide a model')
--print("model_path: ", opt.model)
local checkpoint = torch.load(opt.model)
local batch_size = 1
local fetch = {'rnn_size', 'input_encoding_size', 'drop_prob_lm', 'cnn_proto', 'cnn_model', 'seq_per_img'}
for k,v in pairs(fetch) do 
  opt[v] = checkpoint.opt[v] -- copy over options from model
  --print("opt[v]: ", v, opt[v])
end
local vocab = checkpoint.vocab -- ix -> word mapping
local protos = checkpoint.protos
protos.expander = nn.FeatExpander(opt.seq_per_img) -- class FeatExpander in net_utils.lua
protos.crit = nn.LanguageModelCriterion() -- class LanguageModelCriterion in LanguageModel.lua
protos.lm:createClones() -- reconstruct clones inside the language model
protos.cnn:evaluate()
protos.lm:evaluate()

print("==Start caption service==")

local HelloWorldHandler = class("HelloWorldHandler", turbo.web.RequestHandler)
local Feed2ModelHandler = class("Feed2ModelHandler", turbo.web.RequestHandler)

function HelloWorldHandler:get()
  self:write("Hello World!")
end

function Feed2ModelHandler:post()
  local img_path = self:get_argument("img_path", "/Users/willwywang-NB/github/site/uploads/178706690477497407.jpg")
  local img_batch_raw = torch.ByteTensor(batch_size, 3, 256, 256)
  local max_index = 1
  local wrapped = false
  local infos = {}
  local img = image.load(img_path, 3, 'byte')
  --print("img: ", img)
  img_batch_raw[1] = image.scale(img, 256, 256) ----- from 0
  --print("img_batch_raw[1]: ", img_batch_raw[1])
  img_batch_raw = net_utils.prepro(img_batch_raw, false, opt.gpuid >= 0) -- preprocess in place, and don't augment
  local feats = protos.cnn:forward(img_batch_raw)
  --print("feats: ", feats)
  local sample_opts = { sample_max = opt.sample_max, beam_size = opt.beam_size, temperature = opt.temperature }
  local seq = protos.lm:sample(feats, sample_opts) -- 这里做出预测
  local sents = net_utils.decode_sequence(vocab, seq)  -- 这里得到句子
  --print("sents: ", sents[1])
  self:write("Caption: " .. sents[1])
end

turbo.web.Application({
  {"/hello", HelloWorldHandler},
  {"/feed_to_model", Feed2ModelHandler}
}):listen(8888)
turbo.ioloop.instance():start()
