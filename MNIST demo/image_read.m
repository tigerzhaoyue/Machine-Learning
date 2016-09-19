clc,clear all;
%I=cell(1,10);  %
caffe.set_mode_gpu();
caffe.set_device(0);
solver=caffe.Solver('./lenet_solver_matlab.prototxt');
net=solver.net;
data_blob=zeros(28,28,3,10);
data_blob=single(data_blob);
label_blob=zeros(1,1,1,10);
label_blob=single(label_blob);
loss_value=zeros(5000,4);
for k=1:4
for j=1:5000
	iternum=strcat(num2str(j),'.bmp');
	for i=0:9
    		folder=strcat('./trainimage/',num2str(i));
    		folder=strcat(folder,'/');
            		imgname=strcat(folder,iternum);
    		data_blob(:,:,:,i+1)=caffe.io.load_image(imgname);
    		%A=data_blob(:,:,:,i+1);
    		%data_blob(:,:,:,i+1)=data_blob(:,:,:,i+1)-mean(A(:));
            data_blob(:,:,:,i+1)= data_blob(:,:,:,i+1)/256;
    		label_blob(1,1,:,i+1)=i;
            %I{i+1}=imread(imgname);
            %subplot(2,5,i+1),imshow(I{i+1});   	
    end
	net.blobs('data').set_data(data_blob);
    	net.blobs('label').set_data(label_blob);
    	solver.step(1);
    	disp(solver.iter());
    	L=net.blobs('loss').get_data();
        	disp(L);
        	loss_value1(j,k)=L;
end
end