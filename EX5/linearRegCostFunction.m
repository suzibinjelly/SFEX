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

theta_0=[0;theta(2:end)];
J=1/(2*m)*((X*theta-y)'*(X*theta-y)+lambda*theta_0'*theta_0);
grad=1/m*(X'*(X*theta-y)+lambda*theta_0);

grad = grad(:);

end