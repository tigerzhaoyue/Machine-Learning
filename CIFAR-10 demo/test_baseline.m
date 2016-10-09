if ~exist('cifar_data','var');
    cifar_data = [];
    cifar_labels = [];
    for ii = 1 : 5
        load(strcat('data_batch_',num2str(ii),'.mat'));
        cifar_data = [cifar_data;data];
        cifar_labels = [cifar_labels;labels];
    end
    meanmat = mean(cifar_data,1);
end
load('test_batch.mat');
caffe.reset_all;
caffe.set_mode_gpu;
caffe.set_device(0);

net = caffe.get_net('cifar10_full_train_test.prototxt','test');
net.copy_from('82base_reverse_iter_120000.caffemodel');

data_shape = net.blobs('data').shape;
batch_size = data_shape(4);

batch_data = zeros(data_shape,'single');
batch_labels = zeros(1,1,1,batch_size,'single');

data_pointer = 1;
pred_result = labels*0;
while true
    data_end = min(data_pointer+batch_size-1,10000);
    data_count = data_end - data_pointer + 1;
    if data_count == 0
        break;
    end
    for ii = 1 : data_count
        batch_data(:,:,:,ii) = reshape(single(data(data_pointer+ii-1,:))-meanmat,32,32,3);
    end
    
    net.blobs('data').set_data(batch_data);
    net.forward_prefilled;
    response = net.blobs('ip1').get_data;
    [~,pred] = max(response,[],1);
    pred = pred' - 1;
    
    pred_result(data_pointer:data_end) = pred(1:data_count);
    data_pointer = data_end + 1;
end

num_correct = numel(find(pred_result == labels));

disp(num_correct/10000);

