require 'nn'
require 'nngraph'

local LSTM = {} -- 这是一个通用模块, 同char-rnn中的一样
-- ref: http://apaszke.github.io/lstm-explained.html
-- ref: https://github.com/wojciechz/learning_to_execute
-- ref: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
function LSTM.lstm(input_size, output_size, rnn_size, n, dropout) -- 512, 9567+1, 512, 1, 0.5
  dropout = dropout or 0 

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- indices giving the sequence of symbols, network input
                                        -- nn.Identity()相当于placeholder的作用, nn.Identity()()是graph node, nn.Identity()是nn.Module, nn.Identity()()是dummy input nodes that perform the identity operation, Identity modules will just copy whatever we provide to the network into the graph.
  for L = 1,n do -- 1
    table.insert(inputs, nn.Identity()()) -- prev_c[L], cell state at time t-1 
    table.insert(inputs, nn.Identity()()) -- prev_h[L], hidden state at time t-1
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_c = inputs[L*2] -- prev_c = inputs[2] (80,512)
    local prev_h = inputs[L*2+1] -- prev_h = inputs[3] (80,512)
    -- the input to this layer
    if L == 1 then 
      x = inputs[1] -- network input (80,512)
      input_size_L = input_size -- 512
    else 
      x = outputs[(L-1)*2] -- 多层LSTM才有用
      if dropout > 0 then x = nn.Dropout(dropout)(x):annotate{name='drop_' .. L} end -- apply dropout, if any
      input_size_L = rnn_size
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L} -- input to hidden, annotate用于加标签, 可视化用的,  L是第几层LSTM
    -- 把(80,512)的input_x线性变换成(80,4*512) 
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L} -- hidden to hidden
    -- 把(80,512)的prev_h线性变换成(80,4*512)
    local all_input_sums = nn.CAddTable()({i2h, h2h}) -- i2h + h2h, 又叫preactivations
    -- element-wise相加后还是(80,4*512)
    -- nngraph重写了module的(), nn.Linear()构造了一个module， 第二个()把他变成一个graph node, 参数是这个node的parent

    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums) -- rnn_size=512 把all_input_sums分成4份
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4) -- nn.SplitTable(2)(reshaped) 是把reshaped按照第二维切分, 得到4个512维的向量
    -- 把all_input_sums拆分之后得到4个(80,512)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)     --(80,512)
    local forget_gate = nn.Sigmoid()(n2) --(80,512)
    local out_gate = nn.Sigmoid()(n3)    --(80,512)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)   --(80,512)

    -- perform the LSTM update
    local next_c           = nn.CAddTable()({  -- next cell state, 得到next_cell_state
        nn.CMulTable()({forget_gate, prev_c}), -- previous cell state contribution, 得到current_forget
        nn.CMulTable()({in_gate,     in_transform})  -- input contribution, 得到current_input
      }) -- next_c 也是(80,512)

    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)}) --把next_c变换后, 与out_gate得到next_hidden_state
    --next_h (80,512)
    
    table.insert(outputs, next_c) -- 输出1: next_cell_state
    table.insert(outputs, next_h) -- 输出2: next_hidden_state
  end

  -- set up the decoder, 把outputs列表的最后一个next_h取出来, 使用dropout处理, 作为最后的输出。如果只有1层LSTM, next_h就是top_h
  local top_h = outputs[#outputs] -- top_h (80,512)
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h):annotate{name='drop_final'} end
  -- dropout理解: 对输入按照binomial采样, 并对采样后的向量rescale, 比如dropout rate=0.5, 输入[1,2,3,4] 采样后变成[1,0,0,4], rescale后变成[2,0,0,8]
  local proj = nn.Linear(rnn_size, output_size)(top_h):annotate{name='decoder'} -- 把512维变换成9567+1
  -- 线性变换成(80,9567+1)
  local logsoft = nn.LogSoftMax()(proj) -- 加上LogSoftMax, 使之具有概率分布意义, 9567+1中概率最大的就是预测出来的下一个单词
  table.insert(outputs, logsoft) -- 输出3: 对next_hidden_state变换后得到的logsoft

  return nn.gModule(inputs, outputs) -- nn.gModule返回一个nn.Module, inputs是table of input nodes, outpus是table of output nodes
                                     -- packs the graph into a covenient module with standard API (:forward(), :backward())
                                     -- inputs: input(80,512), cell_state_t-1(80,512), hidden_state_t-1(80,512)
                                     -- outputs: next_cell_state(80,512), next_hidden_state(80,512), logsoft(80,9568)
end

return LSTM

