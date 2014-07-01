function [ TestData, TestLabel ] = Load_CIFAR_10_Test_Data(  )
%Load_CIFAR_10_Test_Data 读入CIFAR-10图像数据库测试数据

%% 读取数据并存储到TestData中

S=load('data/test_batch.mat');
TestData=S.data';
TestLabel=S.labels;

end

