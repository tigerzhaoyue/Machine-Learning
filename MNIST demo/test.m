clc;
%I=cell(1,10);  
correct=0;
testnum=0;
for j=1:900
for i=0:9
    		iternum=strcat(num2str(j),'.bmp');
    		folder=strcat('./testimage/',num2str(i));
    		folder=strcat(folder,'/');
            	imgname=strcat(folder,iternum);
    		%I{i+1}=imread(imgname);
    		test_data_blob(:,:,:,i+1)=caffe.io.load_image(imgname);
    		test_label_blob=label_blob;
    		%subplot(2,5,i+1),imshow(I{i+1});   	
            testnum=testnum+1;
	end
net.blobs('data').set_data(test_data_blob);
net.blobs('label').set_data(test_label_blob);
net.forward_prefilled();
prob = net.blobs('ip2').get_data();
[max_prob, index] = max(prob);
disp(index)
	for k=1:10
		if(index(k)==k)
			correct=correct+1;
		end
	end
%disp(prob);
end
disp(correct);
disp(testnum);
disp(correct/testnum);