function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes 
% inputSize - the size N of the input vector
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
% labels - an M x 1 matrix containing the labels corresponding for the input data
%

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize);

numCases = size(data, 2);

groundTruth = full(sparse(labels, 1:numCases, 1));
cost = 0;
% MOD START --------------------------------------
% thetagrad = zeros(numClasses, inputSize);
% MOD END --------------------------------------

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.

M = bsxfun(@minus,theta*data,max(theta*data, [], 1));
% ����ָ��
M = exp(M);
% ����Ȼ�����ĵ���eΪ������
p = bsxfun(@rdivide, M, sum(M));
% ��һ��
cost = -1/numCases * groundTruth(:)' * log(p(:)) + lambda/2 * sum(theta(:) .^ 2);
% ���ۺ���
thetagrad = -1/numCases * (groundTruth - p) * data' + lambda * theta;
% �ݶ�

% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = thetagrad(:);
end
