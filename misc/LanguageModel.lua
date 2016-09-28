require 'nn'
local utils = require 'misc.utils'
local net_utils = require 'misc.net_utils'
local LSTM = require 'misc.LSTM'

-------------------------------------------------------------------------------
-- Language Model core
-------------------------------------------------------------------------------

local layer, parent = torch.class('nn.LanguageModel', 'nn.Module')
function layer:__init(opt)
  parent.__init(self)

  -- options for core network
  self.vocab_size = utils.getopt(opt, 'vocab_size') -- required 9567
  self.input_encoding_size = utils.getopt(opt, 'input_encoding_size') --512
  self.rnn_size = utils.getopt(opt, 'rnn_size') --512
  self.num_layers = utils.getopt(opt, 'num_layers', 1) --1
  local dropout = utils.getopt(opt, 'dropout', 0) --0.5

  -- options for Language Model
  self.seq_length = utils.getopt(opt, 'seq_length') --16

  -- create the core lstm network. note: +1 for both the START and END tokens
  self.core = LSTM.lstm(self.input_encoding_size, self.vocab_size + 1, self.rnn_size, self.num_layers, dropout)
  --                         512                  9567 + 1             512            1                0.5
  self.lookup_table = nn.LookupTable(self.vocab_size + 1, self.input_encoding_size)
  --                                 9567 + 1             512
  self:_createInitState(1) -- will be lazily resized later during forward passes, 在forward()过程中会输入batch_size

  print(":: LanguageModel.init, Finish init LanguageModel")
end

function layer:_createInitState(batch_size) -- batch_size=16*5, 初始化state, init_state作为forward时第一轮迭代的输入, 表示当前伴随输入xt的LSTM网络的cell_state和hidden_state
  assert(batch_size ~= nil, 'batch size must be provided')
  -- construct the initial state for the LSTM
  if not self.init_state then self.init_state = {} end -- lazy init
  for h=1,self.num_layers*2 do -- self.num_layer=1
    -- note, the init state Must be zeros because we are using init_state to init grads in backward call too
    if self.init_state[h] then -- we have init_state[1], init_state[2], 每个都是(batch_size, rnn_size) (16*5, 512), init_state中的两个(80,512)分别表示的是cell_state, hidden_state
      if self.init_state[h]:size(1) ~= batch_size then
        self.init_state[h]:resize(batch_size, self.rnn_size):zero() -- expand the memory
      end
    else
      self.init_state[h] = torch.zeros(batch_size, self.rnn_size) -- 都初始化成0
    end
  end
  self.num_state = #self.init_state
end

function layer:createClones()
  -- construct the net clones
  print('constructing clones inside the LanguageModel')
  self.clones = {self.core} -- core是LSTM, 参数个数, 7009632
  self.lookup_tables = {self.lookup_table} -- 参数个数, (9567+1, 512), 4898816, lookup_table是做embedding的工作, 把单词word映射到512维的空间上, 每一个词对应512个权重
  for t=2,self.seq_length+2 do -- 16+2
    self.clones[t] = self.core:clone('weight', 'bias', 'gradWeight', 'gradBias') -- 共享weight, bias, gradWeight, gradBias
    self.lookup_tables[t] = self.lookup_table:clone('weight', 'gradWeight') -- 共享weight, gradWeight
  end
end

function layer:getModulesList()
  return {self.core, self.lookup_table}
end

function layer:parameters() -- 应该遍历所有的weights和gradWeights, 把它们都单一的tensor中返回
  -- we only have two internal modules, return their params, LM中有两个内部modules
  local p1,g1 = self.core:parameters()
  local p2,g2 = self.lookup_table:parameters()

  local params = {} -- 保存所有的weights
  for k,v in pairs(p1) do 
    print(":: LanguageModel:parameters(), p1 key:", k)
    print(':: LanguageModel:parameters(), total number of parameters of above key: ', v:nElement()) -- 
    table.insert(params, v) 
  end
  for k,v in pairs(p2) do 
    print(":: LanguageModel:parameters(), p2 key:", k)
    print(':: LanguageModel:parameters(), total number of parameters of above key: ', v:nElement()) -- 
    table.insert(params, v) 
  end
  
  local grad_params = {} -- 保存所有的gradWeights
  for k,v in pairs(g1) do table.insert(grad_params, v) end
  for k,v in pairs(g2) do table.insert(grad_params, v) end

  -- todo: invalidate self.clones if params were requested?
  -- what if someone outside of us decided to call getParameters() or something?
  -- (that would destroy our parameter sharing because clones 2...end would point to old memory)

  return params, grad_params 
end

function layer:training()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do 
    v:training() 
  end
  for k,v in pairs(self.lookup_tables) do 
    v:training() 
  end
end

function layer:evaluate()
  if self.clones == nil then self:createClones() end -- create these lazily if needed
  for k,v in pairs(self.clones) do v:evaluate() end
  for k,v in pairs(self.lookup_tables) do v:evaluate() end
end

--[[
takes a batch of images and runs the model forward in sampling mode
Careful: make sure model is in :evaluate() mode if you're calling this.
Returns: a DxN LongTensor with integer elements 1..M, 
where D is sequence length and N is batch (so columns are sequences)
--]]
function layer:sample(imgs, opt)
  local sample_max = utils.getopt(opt, 'sample_max', 1)
  local beam_size = utils.getopt(opt, 'beam_size', 1)
  local temperature = utils.getopt(opt, 'temperature', 1.0)
  if sample_max == 1 and beam_size > 1 then 
    return self:sample_beam(imgs, opt) 
  end -- indirection for beam search

  local batch_size = imgs:size(1)
  self:_createInitState(batch_size)
  local state = self.init_state

  -- we will write output predictions into tensor seq
  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  local logprobs -- logprobs predicted in last time step
  for t=1,self.seq_length+2 do

    local xt, it, sampleLogprobs
    if t == 1 then
      -- feed in the images
      xt = imgs
    elseif t == 2 then
      -- feed in the start tokens
      it = torch.LongTensor(batch_size):fill(self.vocab_size+1)
      xt = self.lookup_table:forward(it)
    else
      -- take predictions from previous time step and feed them in
      if sample_max == 1 then
        -- use argmax "sampling"
        sampleLogprobs, it = torch.max(logprobs, 2)
        it = it:view(-1):long()
      else
        -- sample from the distribution of previous predictions
        local prob_prev
        if temperature == 1.0 then
          prob_prev = torch.exp(logprobs) -- fetch prev distribution: shape Nx(M+1)
        else
          -- scale logprobs by temperature
          prob_prev = torch.exp(torch.div(logprobs, temperature))
        end
        it = torch.multinomial(prob_prev, 1)
        sampleLogprobs = logprobs:gather(2, it) -- gather the logprobs at sampled positions
        it = it:view(-1):long() -- and flatten indices for downstream processing
      end
      xt = self.lookup_table:forward(it)
    end

    if t >= 3 then 
      seq[t-2] = it -- record the samples
      seqLogprobs[t-2] = sampleLogprobs:view(-1):float() -- and also their log likelihoods
    end

    local inputs = {xt,unpack(state)}
    local out = self.core:forward(inputs)
    logprobs = out[self.num_state+1] -- last element is the output vector
    state = {}
    for i=1,self.num_state do table.insert(state, out[i]) end
  end

  -- return the samples and their log likelihoods
  return seq, seqLogprobs
end

--[[
Implements beam search. Really tricky indexing stuff going on inside. 
Not 100% sure it's correct, and hard to fully unit test to satisfaction, but
it seems to work, doesn't crash, gives expected looking outputs, and seems to 
improve performance, so I am declaring this correct.
]]--
function layer:sample_beam(imgs, opt)
  local beam_size = utils.getopt(opt, 'beam_size', 10)
  local batch_size, feat_dim = imgs:size(1), imgs:size(2)
  local function compare(a,b) return a.p > b.p end -- used downstream

  assert(beam_size <= self.vocab_size+1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed')

  local seq = torch.LongTensor(self.seq_length, batch_size):zero()
  local seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
  -- lets process every image independently for now, for simplicity
  for k=1,batch_size do

    -- create initial states for all beams
    self:_createInitState(beam_size)
    local state = self.init_state

    -- we will write output predictions into tensor seq
    local beam_seq = torch.LongTensor(self.seq_length, beam_size):zero()
    local beam_seq_logprobs = torch.FloatTensor(self.seq_length, beam_size):zero()
    local beam_logprobs_sum = torch.zeros(beam_size) -- running sum of logprobs for each beam
    local logprobs -- logprobs predicted in last time step, shape (beam_size, vocab_size+1)
    local done_beams = {}
    for t=1,self.seq_length+2 do

      local xt, it, sampleLogprobs
      local new_state
      if t == 1 then
        -- feed in the images
        local imgk = imgs[{ {k,k} }]:expand(beam_size, feat_dim) -- k'th image feature expanded out
        xt = imgk
      elseif t == 2 then
        -- feed in the start tokens
        it = torch.LongTensor(beam_size):fill(self.vocab_size+1)
        xt = self.lookup_table:forward(it)
      else
        --[[
          perform a beam merge. that is,
          for every previous beam we now many new possibilities to branch out
          we need to resort our beams to maintain the loop invariant of keeping
          the top beam_size most likely sequences.
        ]]--
        local logprobsf = logprobs:float() -- lets go to CPU for more efficiency in indexing operations
        ys,ix = torch.sort(logprobsf,2,true) -- sorted array of logprobs along each previous beam (last true = descending)
        local candidates = {}
        local cols = math.min(beam_size,ys:size(2))
        local rows = beam_size
        if t == 3 then rows = 1 end -- at first time step only the first beam is active
        for c=1,cols do -- for each column (word, essentially)
          for q=1,rows do -- for each beam expansion
            -- compute logprob of expanding beam q with word in (sorted) position c
            local local_logprob = ys[{ q,c }]
            local candidate_logprob = beam_logprobs_sum[q] + local_logprob
            table.insert(candidates, {c=ix[{ q,c }], q=q, p=candidate_logprob, r=local_logprob })
          end
        end
        table.sort(candidates, compare) -- find the best c,q pairs

        -- construct new beams
        new_state = net_utils.clone_list(state)
        local beam_seq_prev, beam_seq_logprobs_prev
        if t > 3 then
          -- well need these as reference when we fork beams around
          beam_seq_prev = beam_seq[{ {1,t-3}, {} }]:clone()
          beam_seq_logprobs_prev = beam_seq_logprobs[{ {1,t-3}, {} }]:clone()
        end
        for vix=1,beam_size do
          local v = candidates[vix]
          -- fork beam index q into index vix
          if t > 3 then
            beam_seq[{ {1,t-3}, vix }] = beam_seq_prev[{ {}, v.q }]
            beam_seq_logprobs[{ {1,t-3}, vix }] = beam_seq_logprobs_prev[{ {}, v.q }]
          end
          -- rearrange recurrent states
          for state_ix = 1,#new_state do
            -- copy over state in previous beam q to new beam at vix
            new_state[state_ix][vix] = state[state_ix][v.q]
          end
          -- append new end terminal at the end of this beam
          beam_seq[{ t-2, vix }] = v.c -- c'th word is the continuation
          beam_seq_logprobs[{ t-2, vix }] = v.r -- the raw logprob here
          beam_logprobs_sum[vix] = v.p -- the new (sum) logprob along this beam

          if v.c == self.vocab_size+1 or t == self.seq_length+2 then
            -- END token special case here, or we reached the end.
            -- add the beam to a set of done beams
            table.insert(done_beams, {seq = beam_seq[{ {}, vix }]:clone(), 
                                      logps = beam_seq_logprobs[{ {}, vix }]:clone(),
                                      p = beam_logprobs_sum[vix]
                                     })
          end
        end
        
        -- encode as vectors
        it = beam_seq[t-2]
        xt = self.lookup_table:forward(it)
      end

      if new_state then state = new_state end -- swap rnn state, if we reassinged beams

      local inputs = {xt,unpack(state)}
      local out = self.core:forward(inputs)
      logprobs = out[self.num_state+1] -- last element is the output vector
      state = {}
      for i=1,self.num_state do table.insert(state, out[i]) end
    end

    table.sort(done_beams, compare)
    seq[{ {}, k }] = done_beams[1].seq -- the first beam has highest cumulative score
    seqLogprobs[{ {}, k }] = done_beams[1].logps
  end

  -- return the samples and their log likelihoods
  return seq, seqLogprobs
end

--[[
input is a tuple of:
1. torch.Tensor of size NxK (K is dim of image code), N=16*5, K=512
2. torch.LongTensor of size DxN, elements 1..M,       D=16,   N=16*5
   where M = opt.vocab_size and D = opt.seq_length

returns a (D+2)xNx(M+1) Tensor giving (normalized) log probabilities for the 
next token at every iteration of the LSTM (+2 because +1 for first dummy 
img forward, and another +1 because of START/END tokens shift)

输出是(seq_length+2, 16图*5句, 9567+1) 

--]]
function layer:updateOutput(input)
  local imgs = input[1] -- (16*5,512) 16个图,每个图扩展5个(和5个caption对齐),一个图512维图像特征
  local seq = input[2] -- (16,16*5)  每个caption 16个单词, 16个图*每个图5个caption
  if self.clones == nil then self:createClones() end -- lazily create clones on first forward pass

  assert(seq:size(1) == self.seq_length) -- 确认seq的第一个维度是16, 跟seq_length相同, 表示每个caption的长度
  local batch_size = seq:size(2) -- 16*5, 16个图每个图5个caption
  self.output:resize(self.seq_length+2, batch_size, self.vocab_size+1) --(16+2, 16*5, 9567+1)
                --
  
  self:_createInitState(batch_size)

  self.state = {[0] = self.init_state} -- 这样特意让state是从0开始的下标, state[0]就是init_state, 而init_state是两个(80,512)的table, 表示的是cell_state和hidden_state
  self.inputs = {}
  self.lookup_tables_inputs = {}
  self.tmax = 0 -- we will keep track of max sequence length encountered in the data for efficiency
  for t=1,self.seq_length+2 do

    local can_skip = false
    local xt
    if t == 1 then
      -- feed in the images, 这一轮循环只是初始化了xt
      xt = imgs -- NxK sized input, (16*5, 512), 80个图, 每个图512维图像特征, 作为输入到LM模型的第一个输入
    elseif t == 2 then
      -- feed in the start tokens
      local it = torch.LongTensor(batch_size):fill(self.vocab_size+1) --batch_size=16*5, 每个都用9568来填充, 表示9568表示START/END, it是(80,1)
      self.lookup_tables_inputs[t] = it -- lookup_tables_inputs[2] = it
      xt = self.lookup_tables[t]:forward(it) -- NxK sized input (token embedding vectors), lookup_table (9567+1, 512)
      -- it是一个1维Tensor, 长度80, it=[9568,...,9568], 输入到lookup_tables[2]的forward中, lookup_tables[2]是(9568, 512), 会返回(80,512)的tensor, 这80个512是维元素相同的, 每个都是lookup_table[2]的第9568号元素. xt是(80, 512), 表示的是t时刻的80个词对应的512维input encoding, 在t=2时, 实际上是START/END的512为input encoding
    else
      -- feed in the rest of the sequence...
      local it = seq[t-2]:clone() --seq是(16词, 16图*5句), seq[t-2]:clone()得到的是16图*5句的第t-2个词, it是(16*5,1)
      if torch.sum(it) == 0 then -- 表示没词了, 不需要再forward了
        -- computational shortcut for efficiency. All sequences have already terminated and only
        -- contain null tokens from here on. We can skip the rest of the forward pass and save time
        can_skip = true 
      end
      --[[
        seq may contain zeros as null tokens, make sure we take them out to any arbitrary token
        that won't make lookup_table crash with an error.
        token #1 will do, arbitrarily. This will be ignored anyway
        because we will carefully set the loss to zero at these places
        in the criterion, so computation based on this value will be noop for the optimization.
      --]]
      it[torch.eq(it,0)] = 1 -- 会把it中所有是0的换成1, 1表示的是arbitrary token

      if not can_skip then
        self.lookup_tables_inputs[t] = it -- it, (16*5,1)表是的是t时刻的输入词对应的token
        xt = self.lookup_tables[t]:forward(it) -- xt是(80,512), 表示的是t时刻80个词对应的各自512维input encoding
      end
    end

    if not can_skip then
      -- construct the inputs
      self.inputs[t] = {xt,unpack(self.state[t-1])} -- self.state[t-1] 是两个(80,512), 表示的是prev_c, prev_h, xt也是个(80,512), 最终的inputs[t]是[xt, prev_c, prev_h]
      -- forward the network, 把t时刻的输入, 喂给t时刻对应的LSTM子层
      local out = self.clones[t]:forward(self.inputs[t]) -- 因为clones中都是LSTM, 这里实际调用了LSTM
                                                         -- 输入给LSTM[t]的是{input(80,512),prev_c(80,512), prev_h(80,512)}
                                                         -- 输出out是[cell_state, hidden_state, logsoft]
      -- process the outputs
      self.output[t] = out[self.num_state+1] -- last element is the output vector, num_state=2, out[3]是LM返回的logsoft, (80, 9568), 把logsoft放到最终的输出中
      self.state[t] = {} -- the rest is state, 先清空state
      for i=1,self.num_state do 
        table.insert(self.state[t], out[i]) -- 把LSTM t子层的输出隐状态[cell_state, hidden_state]更新到state, 作为下一个LSTM子层的输入
      end
      self.tmax = t -- 记录下forward过程真正处理了几个单词, t最大不会超过seq_length(即16)
    end
  end -- end of for loop

  return self.output -- (18,80,9568)
end

--[[
gradOutput is an (D+2)xNx(M+1) Tensor.
--]]
function layer:updateGradInput(input, gradOutput) -- input是{expanded_feats, data.labels}这个input这里没有用,只是保留形式上的统一(doc中说backward要输入跟forward一样的input,此步中真正有用的input在self.inputs中), gradOutput是dlogprobs(18.80,9568), 是protos.crit的backward的输出
  local dimgs -- grad on input images

  -- go backwards and lets compute gradients
  local dstate = {[self.tmax] = self.init_state} -- this works when init_state is all zeros
  for t=self.tmax,1,-1 do
    -- concat state gradients and output vector gradients at time step t
    local dout = {}
    for k=1,#dstate[t] do  -- dstate[t]有2个元素, 分别表示d_cell_state, d_hidden_state, 放入dout
      table.insert(dout, dstate[t][k]) 
    end
    table.insert(dout, gradOutput[t]) -- dout中再加入gradOutput[t]

    local dinputs = self.clones[t]:backward(self.inputs[t], dout) -- inputs[t]跟forward()中喂给LSTM[t]的输入一样[xt, cell_state, hidden_state], dout则是{d_cell_state, d_hidden_state, gradOutput[t]} , gradOutput[t]是(80,9568)
    -- 这里dout是{d_cell_state, d_hidden_state, gradOutput[t]}的原因是LSTM[t]被定义为nngraph, 拥有了标准的forward和backward, 以及确定的输入和输出, 输入是input(80,512), cell_state_t-1(80,512), hidden_state_t-1(80,512), 输出是next_cell_state(80,512), next_hidden_state(80,512), logsoft(80,9568), 看doc标准backward的定义是[gradInput] backward(input, gradOutput), gradOutput表示gradient w.r.t the output of the module, 即与输出字段对应的其梯度, 所以dout是如上所示.

    -- split the gradient to xt and to state
    local dxt = dinputs[1] -- first element is the input vector
    dstate[t-1] = {} -- copy over rest to state grad
    for k=2,self.num_state+1 do 
      table.insert(dstate[t-1], dinputs[k])  -- 看样子dinputs的第一个是dxt, 后两个分别是d_cell_state, d_hidden_state
    end
    
    -- continue backprop of xt
    if t == 1 then
      dimgs = dxt
    else
      local it = self.lookup_tables_inputs[t]
      self.lookup_tables[t]:backward(it, dxt) -- backprop into lookup table
    end
  end

  -- we have gradient on image, but for LongTensor gt sequence we only create an empty tensor - can't backprop
  self.gradInput = {dimgs, torch.Tensor()}
  return self.gradInput
end

-------------------------------------------------------------------------------
-- Language Model-aware Criterion
-------------------------------------------------------------------------------

local crit, parent = torch.class('nn.LanguageModelCriterion', 'nn.Criterion')
function crit:__init()
  parent.__init(self)
end

--[[
input is a Tensor of size (D+2)xNx(M+1) (16+2)x80x(9567+1)
seq is a LongTensor of size DxN (16x80). The way we infer the target 
in this criterion is as follows:
- at first time step the output is ignored (loss = 0). It's the image tick
- the label sequence "seq" is shifted by one to produce targets
- at last time step the output is always the special END token (last dimension)
The criterion must be able to accomodate variably-sized sequences by making sure
the gradients are properly set to zeros where appropriate.
--]]
function crit:updateOutput(input, seq)
  self.gradInput:resizeAs(input):zero() -- reset to zeros
  local L,N,Mp1 = input:size(1), input:size(2), input:size(3) -- L=18轮迭代, N=80个图片(caption句子个数, 由称为batch大小), Mp1=9568单词种类综述
  local D = seq:size(1) -- D=16个单词
  assert(D == L-2, 'input Tensor should be 2 larger in time')

  local loss = 0
  local n = 0
  for b=1,N do -- iterate over batches
    local first_time = true
    for t=2,L do -- iterate over sequence time (ignore t=1, dummy forward for the image)

      -- fetch the index of the next token in the sequence
      local target_index
      if t-1 > D then -- we are out of bounds of the index sequence: pad with null tokens
        target_index = 0
      else
        target_index = seq[{t-1,b}] -- t-1 is correct, since at t=2 START token was fed in and we want to predict first word (and 2-1 = 1).
        -- target_index是t-1对应的时刻的单词id
      end
      -- the first time we see null token as next index, actually want the model to predict the END token
      if target_index == 0 and first_time then
        target_index = Mp1
        first_time = false
      end

      -- if there is a non-null next token, enforce loss!
      if target_index ~= 0 then
        -- accumulate loss
        loss = loss - input[{ t,b,target_index }] -- log(p), input是(18,80,9568)表示的是第b个句子的第t个单词在target_index上的概率
        self.gradInput[{ t,b,target_index }] = -1 -- 设置gradInput[t][b][target_index]=-1
        n = n + 1
      end
    end
  end
  self.output = loss / n -- normalize by number of predictions that were made
  self.gradInput:div(n) -- 每个元素都除以n, gradInput是(18,80,9568)的Tensor
  return self.output --这个是loss
end

function crit:updateGradInput(input, seq)
  return self.gradInput
end
