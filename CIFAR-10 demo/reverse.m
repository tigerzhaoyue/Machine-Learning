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
cifar_data = [cifar_data;cifar_data];
cifar_labels = [cifar_labels;cifar_labels];
img=zeros(32,32,3);
for row=1:50000
    index=1;
    for i=1:3
    	for j=1:32
		for k=1:32
			img(j,k,i)=cifar_data(row,index);
			index=index+1;
		end

	end
    end
     img_reverse=flipdim(img,2);
     index=1;
     for i=1:3
        for j=1:32
        for k=1:32
                cifar_data(row+50000,index)=img_reverse(j,k,i);
                index=index+1;
        end
    end
    end
%imshow(uint8(img));
end