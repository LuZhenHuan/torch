require 'csvigo'

mTrain = 110500
mTest = 39000
--trainData
dataOriginal = csvigo.load{path = 'reMITP31.csv',mode = 'raw'}

Data = torch.Tensor(mTrain,400)

for i=1,mTrain do
	for j=1,400 do
		Data[i][j] = dataOriginal[i][j];
	end
end
print(#Data)	

label = torch.Tensor(mTrain,1):fill(1)

label[{{mTrain/2+1,mTrain},1}]=2
--label[{{1,mTrain/2},2}]=0

trainset = {data = Data,label = label}
torch.save('trainset12.t7',trainset)
print('save trainset done')
--testData
dataOriginal = csvigo.load{path = 'reMITP31test.csv',mode = 'raw'}

Data = torch.Tensor(mTest,400)

for i=1,mTest do
	for j=1,400 do
		Data[i][j] = dataOriginal[i][j];
	end
end
print(#Data)	

label = torch.Tensor(mTest,1):fill(1)
label[{{mTest/2+1,mTest},1}]=2
--label[{{1,mTest/2},2}]=0

testset = {data = Data,label = label}
torch.save('testset12.t7',testset)
print('save testset done')

print('label 1 and 2')

