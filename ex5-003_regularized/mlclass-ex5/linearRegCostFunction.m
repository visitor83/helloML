function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Note that should not regularize the theta0 term.
temp = theta;
[row col] =  size (theta);
temp = [zeros(1, col) ;  theta(2:end)];

J = sum ((X * theta  - y) .^ 2) + lambda *  sum (temp .^2);
J = J * 0.5 / m;


grad =  X ' * (X * theta  - y) + lambda * temp;
grad = grad .* (1 / m);









% =========================================================================

grad = grad(:);

end
