function [ TrainData, TrainLabel ] = Load_CIFAR_10_Train_Data( TrainFileNum, TrainPerFile, inputSize )
%Load_CIFAR_10_Train_Data 读入CIFAR-10图像数据库训练数据

%% 初始化矩阵用来存储训练数据

TrainNum=TrainFileNum*TrainPerFile;     % 计算训练图片总数
TrainData=zeros(inputSize,TrainNum);    % 每行一个图片
TrainLabel=zeros(TrainNum,1);           % 列向量，每行对应标签号

%% 读取数据并存储到TrainData中
TrainPreString='data/data_batch_';
TrainSufString='.mat';
for i=1:TrainFileNum
    FileName=strcat(TrainPreString,int2str(i),TrainSufString);
    S=load(FileName);
    colBegin=(i-1)*TrainPerFile+1;
    colEnd=i*TrainPerFile;
    TrainData(:,colBegin:colEnd)=S.data';
    TrainLabel(colBegin:colEnd,:)=S.labels;
end;
end

