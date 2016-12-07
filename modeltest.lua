--modeltest

require 'nn'

model = torch.load('modelMSEdone12.t7')
testset = torch.load('testset12.t7')

accuracy = 0

for i=1,39000 do
	predicted = model:forward(testset.data[i])[1]
	--print (predicted)
	if predicted < 1.5 then predicted = 1
	else predicted = 2 end
	--print (predicted)
	if predicted==testset.label[i][1] then accuracy = accuracy +1 end
	--print(testset.label[i])
	--print(accuracy)
end
	
print (accuracy/39000*100 ..'%')	


