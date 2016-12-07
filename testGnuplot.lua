require 'gnuplot'

month = torch.range(1,10)
price = torch.Tensor{28993,29110,29436,30791,33384,36762,39900,39972,40230,40146}
month_predict = torch.range(1,12)
price_predict = torch.Tensor{ 6881.5442,11965.3117,17049.0792, 22132.8468, 27216.6143, 32300.3818, 37384.1493, 42467.9168, 47551.6843, 52635.4518, 57719.2193,62802.9868}

gnuplot.plot({month,price},{month_predict,price_predict:reshape(12)})
--gnuplot.plot(month_predict,price_predict)
