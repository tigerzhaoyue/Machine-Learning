clc;
I=cell(1,10); 
showimage=0;
correct=0;
testnum=0;
test_label_blob=zeros(1,1,1,10);
test_label_blob=single(test_label_blob);
for j=1:20
for i=0:9
    		iternum=strcat(num2str(j),'.bmp');
    		folder=strcat('./testimage/',num2str(i));
    		folder=strcat(folder,'/');
            	imgname=strcat(folder,iternum);
    		test_data_blob(:,:,:,i+1)=caffe.io.load_image(imgname);
            label_blob(1,1,:,i+1)=i;
            testnum=testnum+1;
            if(showimage)
                  I{i+1}=imread(imgname);
                  subplot(2,5,i+1),imshow(I{i+1}); 
            end
                
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