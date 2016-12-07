--simple Ecg

require 'nn'
require 'torch'  --why?
require 'gnuplot'  --draw a figure
require 'optim'		--new algorithm

--read data (data and label)

classes = {'normal','abnormal'}
trainset = torch.load('trainset12.t7')  -- this name

setmetatable(trainset,{   
    __index = function(t,i)  
        return {t.data[i], t.label[i]}
    end
})
trainset.data = trainset.data:double()

function trainset:size()
    return self.data:size(1)
end


-- build a linear model,linear regression
model = nn.Sequential()
model:add(nn.MulConstant(0.25))  --乘一个常数
model:add(nn.Linear(400,100))
model:add(nn.Sigmoid())
model:add(nn.Linear(100,1))
--model:add(nn.Sigmoid())
--model:add(nn.Linear(2,1))
--model:add(nn.MulConstant(50000))

criterion = nn.MSECriterion()
trainer = nn.StochasticGradient(model,criterion)
-- setting trainer's params
trainer.learningRate = 0.001
trainer.maxIteration = 200
-- train
trainer:train(trainset)

torch.save('modelMSEdone12/4.t7',model)  -- this name must modify
print 'save modelMSEdone12/4 done'






