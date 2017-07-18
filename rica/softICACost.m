
%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACostcopy(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);

%%% YOUR CODE HERE %%%
m = size(x,2);
lambda = params.lambda;
epsilon = params.epsilon;

y = W*x;
m = size(x,2);
L1_norm = sqrt((W*x).^2+epsilon);
L2_norm = norm(W'*(W*x)-x);
cost = lambda * sum(sum(L1_norm)) + 0.5*L2_norm^2/m;
Wgrad = lambda * (W*x)./L1_norm * x' + (W*W'*(W*x)*x'+ (W*x)*(W*x)'*W - 2*W*x*x')/m;


% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);
end