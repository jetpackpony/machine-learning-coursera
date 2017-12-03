function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
h = (exp(X * theta * -1) + 1) .^ -1;
pos = h(find(y == 1), 1);
neg = 1 - h(find(y == 0), 1);
cost = -1 * log([pos; neg]);
theta1toN = [0; theta(2:end,:)];
J = sum(cost) / m + lambda / (2 * m) * sum(theta1toN .^ 2);
grad = (X' * (h - y)) / m + lambda / m * theta1toN;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
