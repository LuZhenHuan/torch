--month and price

require 'nn'
require 'torch'  --why?
require 'gnuplot'  --draw a figure
require 'optim'		--new algorithm

month = torch.range(1,10)
price = torch.Tensor{28993,29110,29436,30791,33384,36762,39900,39972,40230,40146}

-- build a linear model,linear regression
model = nn.Sequential()
model:add(nn.MulConstant(0.1))  --乘一个常数
model:add(nn.Linear(1,3))
model:add(nn.Sigmoid())
model:add(nn.Linear(3,3))
model:add(nn.Sigmoid())
model:add(nn.Linear(3,1))
model:add(nn.MulConstant(50000))

--metric    evaluation criterion MSE
criterion = nn.MSECriterion()

--reshape  why 10,1 ?
month_train = month:reshape(10,1)
price_train = price:reshape(10,1)

gnuplot.figure()

w, dl_dw = model:getParameters()  --w是model里面所有可调参数的集合，dl_dw是每个参数对loss的偏导数
--print (w)

--function feval
feval = function(w_new)  
   if w ~= w_new then w:copy(w_new) end  
    dl_dw:zero()  
  
    price_predict = model:forward(month_train)  
    loss = criterion:forward(price_predict, price_train)  
    model:backward(month_train, criterion:backward(price_predict, price_train))  
    return loss, dl_dw  
end  

params = {  
   learningRate = 1e-2  
}  
  
for i=1,10000 do  
   optim.rprop(feval, w, params)  --新的梯度下降算法
  
   if i%100==0 then  
      gnuplot.plot({month, price}, {month_train:reshape(10), price_predict:reshape(10)})  
   end  
end  


month_predict = torch.range(1,12)
local price_predict = model:forward(month_predict:reshape(12,1))
print(price_predict)

--gnuplot.pngfigure('plot.png')
gnuplot.plot({month,price},{month_predict,price_predict:reshape(12)})
gnuplot.plotflush()




