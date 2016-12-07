-- design model
require('nn')
local model = nn.Sequential()
model:add(nn.SpatialConvolutionMM(3,4,5,5))

-- reset weights
local method = 'xavier'
local model_new = require('weight-init')(model, method)
