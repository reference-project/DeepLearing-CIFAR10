%% 基于深度神经网络对CIFAR-10图像数据库进行线性分类
%
%
% * 电子信息工程学院 2011级 物联网工程1班
% * 孙汉卿
% * 2014年6月24日
%

%% STEP 0：初始化参数
clear all;
%%
% Matlab参数
DEBUG = false;                  % 是否正在调试
TEST=false;
addpath func/;
addpath minFunc/;
%%
% 外部参数初始化
TrainFileNum=5;                 % 5个输入文件
if TEST
    TrainFileNum=1;
end;
TrainPerFile=10000;             % 每个输入文件10,000幅图像
TestNum=10000;                  % 测试数量
inputSize = 32 * 32 * 3;        % 输入矩阵形式 (32x32 RGB)
numClasses = 10;            	% 分类数量
load('data\batches.meta.mat');  % label_names矩阵
%%
% 神经网络初始化
hiddenSizeL1 = 300;             % Layer 1 Hidden Size
hiddenSizeL2 = 300;             % Layer 2 Hidden Size
hiddenSizeL3 = 300;             % Layer 3 Hidden Size
sparsityParam = 0.1;            % 平均激活度
lambda = 3e-3;                  % 权重衰减
beta = 3;                       % 惩罚项权重
kPCA=502;
if TEST
    kPCA=366;
end;
options = struct;
options.Method = 'lbfgs';
options.maxIter = 400;
if TEST
    options.maxIter = 2;
end;
options.display = 'on';
softmaxLambda = 1e-4;
softoptions = struct;
softoptions.maxIter = 400;
if TEST
    softoptions.maxIter = 2;
end;

%% STEP 1：读入数据
%  数据库存储在当前目录的data文件夹下，如下命名：
%  batches.meta.mat             % 存储了标签名 label_names
%  data_batch_1.mat             % 训练数据 data labels batch_label，每组10,000个32×32的RGB图像
%  data_batch_2.mat
%  data_batch_3.mat
%  data_batch_4.mat
%  data_batch_5.mat
%  test_batch.mat               % 测试数据 data labels batch_label，含10,000个32×32的RGB图像
%%
% 训练数据
[ trainData, trainLabels ] = Load_CIFAR_10_Train_Data(TrainFileNum, TrainPerFile, inputSize);
trainLabels=trainLabels+1;      % 0-9的标签转为1-10标签
%%
% 标签名数据
load('data\batches.meta.mat');  % label_names矩阵
%%
% 判断数据是否正常
if ~DEBUG
    assert(size(label_names,1)==numClasses,'标签名数据读取出错：data\batches.meta.mat');
end;

%% STEP 2：数据预处理
% 进行PCA，并存储了相应的参数
%% 
% 均值标准化为0
avg=mean(trainData,1);
trainData=trainData-repmat(avg,size(trainData,1),1);
%%
% $\sigma$ 及其特征值
sigma=trainData*trainData'/size(trainData,2);
[U,S,~]=svd(sigma);
%%
% PCA主成分
% xRot=U'*trainData;
trainData=U(:,1:kPCA)' * trainData;
%%
% 输入尺寸改为PCA的输入
inputSize=kPCA;
%%
% 测试时计算误差，确定PCA保留的主成分维度
if TEST
    SS=diag(S);
    PCA=cumsum(SS)./sum(SS);
end;

%% STEP 3：训练自编码各层网络参数
%%
% 第一层
sae1Theta = initializeParameters(hiddenSizeL1, inputSize);          % 随机初始化起始点
[sae1OptTheta, ~] =  minFunc(@(p)sparseAutoencoderCost(p,...
    inputSize,hiddenSizeL1,lambda,sparsityParam,beta,trainData),...
    sae1Theta,options);                                             % 训练出第一层网络的参数
save('saves/step1.mat', 'sae1OptTheta');                            % 保存结果
[sae1Features] = feedForwardAutoencoder(sae1OptTheta, hiddenSizeL1, ...
    inputSize, trainData);                                          % 得到一阶特征
%%
% 第二层
sae2Theta = initializeParameters(hiddenSizeL2, hiddenSizeL1);
[sae2OptTheta, ~] =  minFunc(@(p)sparseAutoencoderCost(p,...
    hiddenSizeL1,hiddenSizeL2,lambda,sparsityParam,beta,sae1Features),...
    sae2Theta,options);
save('saves/step2.mat', 'sae2OptTheta');
[sae2Features] = feedForwardAutoencoder(sae2OptTheta, hiddenSizeL2, ...
    hiddenSizeL1, sae1Features);
%%
% 第三层
sae3Theta = initializeParameters(hiddenSizeL3, hiddenSizeL2);
[sae3OptTheta, ~] =  minFunc(@(p)sparseAutoencoderCost(p,...
    hiddenSizeL2,hiddenSizeL3,lambda,sparsityParam,beta,sae1Features),...
    sae3Theta,options);
save('saves/step3.mat', 'sae3OptTheta');
[sae3Features] = feedForwardAutoencoder(sae3OptTheta, hiddenSizeL3, ...
    hiddenSizeL2, sae2Features);

%% STEP 4：Softmax训练
% 使用三阶特征直接训练Softmax分类器
saeSoftmaxTheta = 0.005 * randn(hiddenSizeL3 * numClasses, 1);
softmaxModel = softmaxTrain(hiddenSizeL3,numClasses,softmaxLambda,...
    sae3Features,trainLabels,softoptions);
saeSoftmaxOptTheta = softmaxModel.optTheta(:);
save('saves/step4.mat', 'saeSoftmaxOptTheta');

%% STEP 5：微调参数
% 栈式编码参数微调
stack = cell(3,1);
stack{1}.w = reshape(sae1OptTheta(1:hiddenSizeL1*inputSize), ...
    hiddenSizeL1, inputSize);
stack{1}.b = sae1OptTheta(2*hiddenSizeL1*inputSize+1:2*hiddenSizeL1*inputSize+hiddenSizeL1);
stack{2}.w = reshape(sae2OptTheta(1:hiddenSizeL2*hiddenSizeL1), ...
    hiddenSizeL2, hiddenSizeL1);
stack{2}.b = sae2OptTheta(2*hiddenSizeL2*hiddenSizeL1+1:2*hiddenSizeL2*hiddenSizeL1+hiddenSizeL2);
stack{3}.w = reshape(sae3OptTheta(1:hiddenSizeL3*hiddenSizeL2), ...
    hiddenSizeL3, hiddenSizeL2);
stack{3}.b = sae3OptTheta(2*hiddenSizeL3*hiddenSizeL2+1:2*hiddenSizeL3*hiddenSizeL2+hiddenSizeL3);
% Initialize the parameters for the deep model
[stackparams, netconfig] = stack2params(stack);
stackedAETheta = [ saeSoftmaxOptTheta ; stackparams ];

[stackedAEOptTheta, cost] =  minFunc(@(p)stackedAECost(p,inputSize,hiddenSizeL3,...
    numClasses, netconfig,lambda, trainData, trainLabels),...
    stackedAETheta,options);
save('saves/step5.mat', 'stackedAEOptTheta');


%% STEP 6：测试数据
[ testData, testLabels ] = Load_CIFAR_10_Test_Data(  );
testLabels = testLabels+1;
%% 
% 使用输入数据的参数进行PCA
avg2=mean(testData,1);
testData=double(testData)-repmat(avg2,size(testData,1),1);
% testRot=U'*testData;
testData=U(:,1:inputSize)'*testData;

%% STEP 7：测试
% 进行预测并分析精度
%% 
% 微调前
[pred] = stackedAEPredict(stackedAETheta, inputSize, hiddenSizeL3, ...
    numClasses, netconfig, testData);
acc = mean(testLabels(:) == pred(:));
fprintf('微调前精度：%0.3f%%\n', acc * 100);
%%
% 微调后
[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL3, ...
    numClasses, netconfig, testData);
acc = mean(testLabels(:) == pred(:));
fprintf('微调后精度：%0.3f%%\n', acc * 100);
%%
% 微调后的训练集精度
[pred] = stackedAEPredict(stackedAEOptTheta, inputSize, hiddenSizeL3, ...
    numClasses, netconfig, trainData);
acc = mean(trainLabels(:) == pred(:));
fprintf('\n微调后的训练集精度：%0.3f%%\n', acc * 100);
