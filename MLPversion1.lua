--month and price

require 'nn'
require 'torch'  --why?
require 'gnuplot'  --draw a figure

month = torch.range(1,10)
price = torch.Tensor{28993,29110,29436,30791,33384,36762,39900,39972,40230,40146}

-- build a linear model,linear regression
model = nn.Linear(1,1)

--metric    evaluation criterion MSE
criterion = nn.MSECriterion()

--reshape  why 10,1 ?
month_train = month:reshape(10,1)
price_train = price:reshape(10,1)

--epoch
for i=1,1000 do
	price_predict = model:forward(month_train)	--model forward
	--err = criterion:forward(price_predict,price_train)	--criterion
	--print(i,err)
	model:zeroGradParameters()	-- reset the parameters 
	gradient = criterion:backward(price_predict,price_train)
	model:backward(month_train,gradient)
	model:updateParameters(0.01)
end

month_predict = torch.range(1,12)
local price_predict = model:forward(month_predict:reshape(12,1))
print(price_predict)

--gnuplot.pngfigure('plot.png')
gnuplot.plot({month,price},{month_predict,price_predict:reshape(12)})
gnuplot.plotflush()






