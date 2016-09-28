local utils = require 'misc.utils'
local net_utils = {} --lua可以使用使用table定义函数

-- take a raw CNN from Caffe and perform surgery. Note: VGG-16 SPECIFIC!
function net_utils.build_cnn(cnn, opt)
  local layer_num = utils.getopt(opt, 'layer_num', 38)
  print(":: net_utils.build_cnn, layer_num: ", layer_num)
  local backend = utils.getopt(opt, 'backend', 'cudnn')
  local encoding_size = utils.getopt(opt, 'encoding_size', 512)
  
  if backend == 'cudnn' then
    require 'cudnn'
    backend = cudnn
  elseif backend == 'nn' then
    require 'nn'
    backend = nn
  else
    error(string.format('Unrecognized backend "%s"', backend))
  end

  -- copy over the first layer_num layers of the CNN
  local cnn_part = nn.Sequential()
  for i = 1, layer_num do
    local layer = cnn:get(i)

    if i == 1 then
      -- convert kernels in first conv layer into RGB format instead of BGR,
      -- which is the order in which it was trained in Caffe
      local w = layer.weight:clone()
      -- swap weights to R and B channels
      print('converting first layer conv filters from BGR to RGB...')
      -- 有三个通道, 对调1和3通道, BGR->RGB
      layer.weight[{ {}, 1, {}, {} }]:copy(w[{ {}, 3, {}, {} }])
      layer.weight[{ {}, 3, {}, {} }]:copy(w[{ {}, 1, {}, {} }])
    end

    cnn_part:add(layer)
  end

  cnn_part:add(nn.Linear(4096,encoding_size)) --最后增加一层4096->512
  cnn_part:add(backend.ReLU(true)) --cudnn.Relu(true)
  return cnn_part
end

-- takes a batch of images and preprocesses them
-- VGG-16 network is hardcoded, as is 224 as size to forward
function net_utils.prepro(imgs, data_augment, on_gpu)
  assert(data_augment ~= nil, 'pass this in. careful here.') --true
  assert(on_gpu ~= nil, 'pass this in. careful here.') --gpuid=2

  local h,w = imgs:size(3), imgs:size(4) --都是256
  local cnn_input_size = 224 --VGG-16网络写死的

  -- cropping data augmentation, if needed
  --print(":: net_utils.prepro, before crop imgs.size(): ", imgs:size()) -- (16,3,256,256)
  if h > cnn_input_size or w > cnn_input_size then 
    local xoff, yoff
    if data_augment then --true
      xoff, yoff = torch.random(w-cnn_input_size), torch.random(h-cnn_input_size)
    else
      -- sample the center
      xoff, yoff = math.ceil((w-cnn_input_size)/2), math.ceil((h-cnn_input_size)/2)
    end
    -- crop.
    imgs = imgs[{ {}, {}, {yoff,yoff+cnn_input_size-1}, {xoff,xoff+cnn_input_size-1} }]
  end
  --print(":: net_utils.prepro, after crop imgs.size(): ", imgs:size()) -- (16,3,224,224) 改变了img的长和宽

  -- ship to gpu or convert from byte to float
  if on_gpu then imgs = imgs:cuda() else imgs = imgs:float() end
  -- imgs:cuda(): Transfering a Tensor imgs to the GPU:

  -- lazily instantiate vgg_mean
  if not net_utils.vgg_mean then
    net_utils.vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(1,3,1,1) -- in RGB order
    --print(":: net_utils.prepro, init net_utils.vgg_mean", net_utils.vgg_mean)
  end
  net_utils.vgg_mean = net_utils.vgg_mean:typeAs(imgs) -- a noop if the types match
  -- 经过typeAs(imgs)后, net_utils.vgg_mean的类型由torch.FloatTensor变成torch.CudaTensor
  --print(":: net_utils.prepro, after typsAs(imgs) net_utils.vgg_mean", net_utils.vgg_mean)

  -- subtract vgg mean
  imgs:add(-1, net_utils.vgg_mean:expandAs(imgs))
  -- multiply net_tuils.vgg_mean:expandsAs(imgs) with -1 and add to imgs
  -- vgg_mean:expandsAs(imgs)根据imgs的shape扩展vgg_mean
  -- imgs是(16,3,224,224)的, vgg_mean是(1,3,1,1), 扩展后vgg_mean会变成(16,3,224,224), 变成16个图, 每个图的三个色域的(224,224)像素分别由vgg_mean的色域填充
  -- 如果有不清楚的可以运行如下code观察
  -- imgs=torch.Tensor(2,3,5,5)
  -- vgg_mean = torch.FloatTensor{123.68, 116.779, 103.939}:view(1,3,1,1)
  -- vgg_mean = vgg_mean:typeAs(imgs)
  -- vgg_mean:expandAs(imgs)
  -- 直观理解就是把所有的图片的(224,224)区域的RGB色域都减去vgg_mean中的RGB值, img.R - vgg_mean.R, img.G - vgg_mean.G, img.B - vgg_mean.B
  -- 这个vgg_mean对应的颜色是类似土色

  --print(":: net_utils.prepro, final imgs.size(): ", imgs:size())
  return imgs
end

---------Class FeatExpander-------------------------------

-- layer that expands features out so we can forward multiple sentences per image
local layer, parent = torch.class('nn.FeatExpander', 'nn.Module')
function layer:__init(n) -- seq_per_img, 5
  parent.__init(self)
  self.n = n
end

function layer:updateOutput(input) --实际上做的是forward()工作, 把输出写到self.output
  if self.n == 1 then  -- 如果n=1, 那么不需要扩展, 直接输出即可
    self.output = input; 
    return self.output 
  end -- act as a noop for efficiency

  -- simply expands out the features. Performs a copy information
  assert(input:nDimension() == 2) --(16,512)
  local d = input:size(2) -- 512
  self.output:resize(input:size(1)*self.n, d) -- (16*5, 512) 16个图片, 每个图片5个caption
  for k=1,input:size(1) do -- input:size(1) 表示这个batch有多少图片, 16个
    local j = (k-1)*self.n+1 --self.n是5, 表示5个caption
    self.output[{ {j,j+self.n-1} }] = input[{ {k,k}, {} }]:expand(self.n, d) -- copy over, 做的事情是把(16,512)变成(16*5,512), 把原来的每个图片对应的512图像特征复制5份
    -- output的1到5(包含了5), 把input的[1]按照(5,512)复制
  end
  return self.output
end

function layer:updateGradInput(input, gradOutput)
  if self.n == 1 then 
    self.gradInput = gradOutput; 
    return self.gradInput 
  end -- act as noop for efficiency
  -- add up the gradients for each block of expanded features
  self.gradInput:resizeAs(input) --(16,512)
  local d = input:size(2)
  for k=1,input:size(1) do
    local j = (k-1)*self.n+1
    self.gradInput[k] = torch.sum(gradOutput[{ {j,j+self.n-1} }], 1)
  end
  return self.gradInput
end

----------------------------------------

function net_utils.list_nngraph_modules(g)
  local omg = {}
  for i,node in ipairs(g.forwardnodes) do
      local m = node.data.module
      if m then
        table.insert(omg, m)
      end
   end
   return omg
end

function net_utils.listModules(net)
  -- torch, our relationship is a complicated love/hate thing. And right here it's the latter
  local t = torch.type(net)
  local moduleList
  if t == 'nn.gModule' then
    moduleList = net_utils.list_nngraph_modules(net)
  else
    moduleList = net:listModules()
  end
  return moduleList
end

function net_utils.sanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and m.gradWeight then
      --print('sanitizing gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
      m.gradWeight = nil
    end
    if m.bias and m.gradBias then
      --print('sanitizing gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
      m.gradBias = nil
    end
  end
end

function net_utils.unsanitize_gradients(net)
  local moduleList = net_utils.listModules(net)
  for k,m in ipairs(moduleList) do
    if m.weight and (not m.gradWeight) then
      m.gradWeight = m.weight:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradWeight:nElement())
      --print(m.weight:size())
    end
    if m.bias and (not m.gradBias) then
      m.gradBias = m.bias:clone():zero()
      --print('unsanitized gradWeight in of size ' .. m.gradBias:nElement())
      --print(m.bias:size())
    end
  end
end

--[[
take a LongTensor of size DxN with elements 1..vocab_size+1 
(where last dimension is END token), and decode it into table of raw text sentences.
each column is a sequence. ix_to_word gives the mapping to strings, as a table
--]]
function net_utils.decode_sequence(ix_to_word, seq)
  local D,N = seq:size(1), seq:size(2)
  local out = {}
  for i=1,N do
    local txt = ''
    for j=1,D do
      local ix = seq[{j,i}]
      local word = ix_to_word[tostring(ix)]
      if not word then break end -- END token, likely. Or null token
      if j >= 2 then txt = txt .. ' ' end
      txt = txt .. word
    end
    table.insert(out, txt)
  end
  return out
end

function net_utils.clone_list(lst)
  -- takes list of tensors, clone all
  local new = {}
  for k,v in pairs(lst) do
    new[k] = v:clone()
  end
  return new
end

-- hiding this piece of code on the bottom of the file, in hopes that
-- noone will ever find it. Lets just pretend it doesn't exist
function net_utils.language_eval(predictions, id)
  -- this is gross, but we have to call coco python code.
  -- Not my favorite kind of thing, but here we go
  local out_struct = {val_predictions = predictions}
  utils.write_json('coco-caption/val' .. id .. '.json', out_struct) -- serialize to json (ew, so gross)
  os.execute('./misc/call_python_caption_eval.sh val' .. id .. '.json') -- i'm dying over here
  local result_struct = utils.read_json('coco-caption/val' .. id .. '.json_out.json') -- god forgive me
  return result_struct
end

return net_utils
