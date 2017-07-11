clc,clear,close all
addpath('data');
addpath('tools');
addpath('libsvm')
% addpath(genpath(pwd));
%% Data
load IndiaP;
no_train   = round(size(GroundT,1)*0.04);
% no_train = 410;

% load PaviaU
% no_train   = round(size(GroundT,1)*0.04);
% % no_train = 1711;

% load Salinas
% no_train   = round(size(GroundT,1)*0.002);
% no_train = 108;

[r, s, d] = size(img);
GroundT = GroundT';
%% Parameters
k = 20;     sigma_s = 200;  sigma_r = 0.1;  t = 10;
%% Training Set and Test Set
no_classes = length(unique(GroundT(2,:)));

indexes = train_test_random_new(GroundT(2,:),...
          fix(no_train/no_classes),no_train);

train_labels = GroundT(2,indexes);
Train_class_No = zeros(no_classes,1);
for i =1:16
    Train_class_No(i,1) = length(find(train_labels == i));
end
      
train_indexes = GroundT(:,indexes);
test_indexes = GroundT;
test_indexes(:,indexes) = [];
%% Feature Dimension is Reduced from d to k
Fimg = reshape(img,[r*s d]);
Fimg = imresize(Fimg,[r*s k]);
[fimg] = scale_to_01(Fimg);
fimg = reshape(fimg,[r s k]);
%% Spatial Structure
for i = 1:k
    fimg(:,:,i) = ARM(fimg(:,:,i),sigma_s, sigma_r,t);
end
%% Multi-SVM classifer
fimg = im2vector(fimg);
fimg = fimg';
fimg = double(fimg);

train_samples = fimg(:,train_indexes(1,:))';
train_labels  = train_indexes(2,:)';

test_samples  = fimg(:,test_indexes(1,:))';
test_labels   = test_indexes(2,:)';

[train_samples, M, m] = scale_to_n1p1(train_samples);
fimg = scale_to_n1p1(fimg', M, m);

[Ccv, Gcv, cv, cv_t]=cross_validation_svm(train_labels,train_samples);
% Training using a Gaussian RBF kernel
parameter=sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv,Gcv);
model=svmtrain(train_labels,train_samples,parameter);
% Testing
Result = svmpredict(ones(r*s,1),fimg,model); 
%% Evaluate the Performance
GroundTest = double(test_labels(:,1));
ResultTest = Result(test_indexes(1,:));
[OA,AA,kappa,CA] = confusion(GroundTest,ResultTest);
%% Display the result of ARM_HSI
Final = zeros(1,r*s);
Final(GroundT(1,:)) = Result(GroundT(1,:));
Final = reshape(Final,r,s); 
ARM_HSI = label2color(Final,'india');
% ARM_HSI = label2color(Final,'uni');
figure,imshow(ARM_HSI); 
display([OA, AA, kappa])
