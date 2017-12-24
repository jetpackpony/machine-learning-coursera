function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
regTheta = [0; theta(2:end, :)];
cost = 1 / (2 * m) * sum((X * theta - y) .^ 2);
reg = lambda / (2 * m) * sum(regTheta .^ 2);
J = cost + reg;

g = (1 / m) * (X' * (X * theta - y));
reg = (lambda / m) * regTheta;
grad = g + reg;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================


end
