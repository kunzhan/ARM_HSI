clear
addpath('data')
addpath('tools')
load IndiaP;
% load Salinas;
% load PaviaU;
[r, s, ~] = size(img);
GT = zeros(r*s,1);
GT(GroundT(:,1)) = GroundT(:,2);
GT = reshape(GT,r,s);  
gt = label2color(GT,'india');
figure,
% subplot(121),imshow('IndiaP.jpg')
tem = uint8(255*mat2gray(img(:,:,[20,65,110]))); %  IndiaP Salinas
% tem = uint8(255*mat2gray(img(:,:,[65,20,1]))); %  PaviaU
subplot(121),imshow(tem)
subplot(122),imshow(gt); 
% imwrite(tem,'pc_PaviaU.jpg','quality',100)
% imwrite(gt,'gt_PaviaU.jpg','quality',100)
