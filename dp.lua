-- dp是深度学习的依赖包  
require 'dp'  
  
--[[command line arguments]]--  
  
cmd = torch.CmdLine()  
cmd:text()  
cmd:text('Image Classification using MLP Training/Optimization')  
cmd:text('Example:')  
cmd:text('$> th neuralnetwork.lua --batchSize 128 --momentum 0.5')  
cmd:text('Options:')  
--  学习率以及衰减系数相关的控制参数  
cmd:option('--learningRate', 0.1, 'learning rate at t=0')  
cmd:option('--lrDecay', 'linear', 'type of learning rate decay : adaptive | linear | schedule | none')  
cmd:option('--minLR', 0.00001, 'minimum learning rate')  
cmd:option('--saturateEpoch', 300, 'epoch at which linear decayed LR will reach minLR')  
cmd:option('--schedule', '{}', 'learning rate schedule')  
cmd:option('--maxWait', 4, 'maximum number of epochs to wait for a new minima to be found. After that, the learning rate is decayed by decayFactor.')  
cmd:option('--decayFactor', 0.001, 'factor by which learning rate is decayed for adaptive decay.')  
cmd:option('--maxOutNorm', 1, 'max norm each layers output neuron weights')  
cmd:option('--momentum', 0, 'momentum')  
-- 激活函数，隐藏神经元个数，每次训练的个数  
cmd:option('--activation', 'Tanh', 'transfer function like ReLU, Tanh, Sigmoid')  
cmd:option('--hiddenSize', '{200,200}', 'number of hidden units per layer')  
cmd:option('--batchSize', 32, 'number of examples per batch')  
-- 硬件相关的参数GPU  
cmd:option('--cuda', false, 'use CUDA')  
cmd:option('--useDevice', 1, 'sets the device (GPU) to use')  
  
cmd:option('--maxEpoch', 100, 'maximum number of epochs to run')  
cmd:option('--maxTries', 30, 'maximum number of epochs to try to find a better local minima for early-stopping')  
-- 是否使用dropout方法解决过拟合和batchNorm  
cmd:option('--dropout', false, 'apply dropout on hidden neurons')  
cmd:option('--batchNorm', false, 'use batch normalization. dropout is mostly redundant with this')  
-- 数据集的选取以及相关的标准化方法  
cmd:option('--dataset', 'Mnist', 'which dataset to use : Mnist | NotMnist | Cifar10 | Cifar100')  
cmd:option('--standardize', false, 'apply Standardize preprocessing')  
cmd:option('--zca', false, 'apply Zero-Component Analysis whitening')  
cmd:option('--lecunlcn', false, 'apply Yann LeCun Local Contrast Normalization')  
-- 是否打印进度条和相关的opt参数  
cmd:option('--progress', false, 'display progress bar')  
cmd:option('--silent', false, 'dont print anything to stdout')  
cmd:text()  
-- 解析参数  
opt = cmd:parse(arg or {})  
opt.schedule = dp.returnString(opt.schedule)  
opt.hiddenSize = dp.returnString(opt.hiddenSize)  
if not opt.silent then  
   table.print(opt)-- 是否打印相关的opt  
end  


if opt.dataset == 'Mnist' then  
   ds = dp.Mnist{input_preprocess = input_preprocess}  
elseif opt.dataset == 'NotMnist' then  
   ds = dp.NotMnist{input_preprocess = input_preprocess}  
elseif opt.dataset == 'Cifar10' then  
   ds = dp.Cifar10{input_preprocess = input_preprocess}  
elseif opt.dataset == 'Cifar100' then  
   ds = dp.Cifar100{input_preprocess = input_preprocess}  
elseif opt.dataset == 'FaceDetection' then  
   ds = dp.FaceDetection{input_preprocess = input_preprocess}  
else  
   error("Unknown Dataset")  
end  


local input_preprocess = {}  
if opt.standardize then  
   table.insert(input_preprocess, dp.Standardize())  
end  
if opt.zca then  
   table.insert(input_preprocess, dp.ZCA())  
end  
if opt.lecunlcn then  
   table.insert(input_preprocess, dp.GCN())  
   table.insert(input_preprocess, dp.LeCunLCN{progress=true})  
end  

model = nn.Sequential()  
model:add(nn.Convert(ds:ioShapes(), 'bf')) -- to batchSize x nFeature (also type converts)  

inputSize = ds:featureSize()  
for i,hiddenSize in ipairs(opt.hiddenSize) do  
   model:add(nn.Linear(inputSize, hiddenSize)) -- parameters  
   if opt.batchNorm then  
      model:add(nn.BatchNormalization(hiddenSize))  
   end  
   model:add(nn[opt.activation]())  
   if opt.dropout then  
      model:add(nn.Dropout())  
   end  
   inputSize = hiddenSize  
end  




