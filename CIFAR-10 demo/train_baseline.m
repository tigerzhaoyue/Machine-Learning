
meanmat = mean(cifar_data,1);

caffe.reset_all;
caffe.set_mode_gpu;
caffe.set_device(0);

caffe_solver = caffe.get_solver('cifar10_full_solver.prototxt');
%caffe_solver.restore('82_iter_80000.solverstate');
data_shape = caffe_solver.net.blobs('data').shape;
batch_size = data_shape(4);

batch_data = zeros(data_shape,'single');
batch_labels = zeros(1,1,1,batch_size,'single');
while true
    data_sel = randperm(100000,batch_size);
    batch_labels(:) = cifar_labels(data_sel,1);
    for ii = 1 : batch_size
        sel = data_sel(ii);
        batch_data(:,:,:,ii) = reshape(single(cifar_data(sel,:))-meanmat,32,32,3);
    end
    
    caffe_solver.net.blobs('data').set_data(batch_data);
    caffe_solver.net.blobs('label').set_data(batch_labels);
    caffe_solver.step(1);
    
    iter = caffe_solver.iter;
    
    if mod(iter,100) == 0
        loss = caffe_solver.net.blobs('loss').get_data;
        acc = caffe_solver.net.blobs('accuracy').get_data;
        fprintf('iter:%-10d,loss:%f\tacc:%f\n',iter,loss,acc);
    end
end



