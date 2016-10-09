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
img=zeros(32,32,3);
index=1;
row=100;
for i=1:3
	for j=1:32
		for k=1:32
			img(j,k,i)=cifar_data(row,index);
			index=index+1;
		end

	end

end
%img=flipdim(img,2);
imshow(uint8(img));