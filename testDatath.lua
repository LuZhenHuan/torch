for _,dataset_name in ipairs({"train","valid","test"}) do
  datas=nil
  classes=nil
  path_prefix=os.getenv('HOME').."/data/weibo/"
  th_output_prefix=os.getenv('HOME').."/workspace/torch7/"
  path_surfix=".txt"
  for _,index in ipairs({0,1,2,3,4}) do
    data_n={}
    classes_n={}
    file=io.open(path_prefix..dataset_name..index..path_surfix,'r')
    for line in file:lines() do
      line_vector={}
      for element in string.gmatch(line,"%S+") do 
        table.insert(line_vector,element) 
      end
      table.insert(data_n,line_vector)
    end
    data_tensor_n=torch.Tensor(data_n)
    data_tensor_n=data_tensor_n:resize(data_tensor_n:size(1),data_tensor_n:size(2)/100,100)
    classes_tensor_n=torch.Tensor(data_tensor_n:size(1)):fill(index)
    print(data_tensor_n:size())
    print(classes_tensor_n:size())
    datas=datas and torch.cat(datas,data_tensor_n,1) or data_tensor_n
    classes=classes and torch.cat(classes,classes_tensor_n,1) or classes_tensor_n
  end
  classes=classes:int()
  print(datas:size())
  print(classes:size())
  data_object={datas,classes}
  torch.save(th_output_prefix..dataset_name..'.th7',data_object)
end
