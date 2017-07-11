% clc,clear;
% close all;
data='IndiaP';
% data='PaviaU';
% data='Salinas';
% data='Botswana';
% data='KSC';
% data='PaviaC';
addpath(genpath(pwd));
location = [data];
load (location);
percent=4;
numPixels = length(GroundT);
numTrains = numPixels * percent / 100;
switch data
    case{'IndiaP'}
        no_classes = 16;
        no_train= numTrains;
    case{'PaviaC'}
         no_classes = 9;
        no_train= numTrains;
    case{'PaviaU'}
         no_classes = 9;
        no_train= numTrains;
    case{'Salinas'}
        no_classes = 16;
        no_train= numTrains;
    case{'KSC'}
        no_classes = 13;
        no_train= numTrains;
        case{'Botswana'}
        no_classes = 14;
        no_train= numTrains;
    otherwise
        error('not data set');
end

%%%%%
% load the ground truth and the hyperspectral image
[no_lines, no_rows, no_bands] = size(img);
GroundT=GroundT';
%%% construct traing and test dataset
indexes=train_test_random_new(GroundT(2,:),fix(no_train/no_classes),no_train);

k=20;
tic
fimg = reshape(img,[no_lines*no_rows,no_bands]);
fimg = imresize(fimg,[no_lines*no_rows,k]);%'bicubic','nearest'.

[fimg] = scale_to_01(fimg);
fimg = reshape(fimg,[no_lines no_rows k]);
% for i = 1:20
%    figure,imshow(fimg(:,:,i))
% end
k = size(fimg,3);
% for i = 1:k
%    figure,imshow(fimg(:,:,i))
% end

for i = 1:k
    fimg(:,:,i) = ARM(fimg(:,:,i),200, 0.1,10);
end
%%% SVM classification
fimg = im2vector(fimg);
fimg = fimg';
fimg=double(fimg);
%%%
train_SL = GroundT(:,indexes);
train_samples = fimg(:,train_SL(1,:))';
train_labels= train_SL(2,:)';
%
test_SL = GroundT;
test_SL(:,indexes) = [];
test_samples = fimg(:,test_SL(1,:))';
test_labels = test_SL(2,:)';
% Normalizing Training and original img 
[train_samples,M,m] = scale_to_n1p1(train_samples);
[fimg ] = scale_to_n1p1(fimg',M,m);
% Selecting the paramter for SVM
[Ccv Gcv cv cv_t]=cross_validation_svm(train_labels,train_samples);
% Training using a Gaussian RBF kernel
parameter=sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv,Gcv);
model=svmtrain(train_labels,train_samples,parameter);
% Testing
Result = svmpredict(ones(no_lines*no_rows,1),fimg,model); 
% Evaluation
GroudTest = double(test_labels(:,1));
ResultTest = Result(test_SL(1,:),:);
[OA,AA,kappa,CA]=confusion(GroudTest,ResultTest)
toc
% Result = reshape(Result,no_lines,no_rows);
% % VClassMap=label2color(Result,'india');
% VClassMap=label2color(Result,'uni');
% temp = ToVector(VClassMap);
% for i = 1:size(temp,1)
%     if ~any(GroundT(1,:)==i)
%         temp(i,:)=[0 0 0];
%     end
% end
% VClassMap = reshape(temp,no_lines, no_rows, 3);  
% imwrite(VClassMap,'ARMpa.jpg');
% figure,imshow(VClassMap);
% toc