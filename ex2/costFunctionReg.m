function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
b = size(theta); 
grad = zeros(b);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

z = theta'*X';
h_theta = sigmoid(z);
J = (1/m)*(log(h_theta)*-y - log(1 - h_theta)*(1 - y)) + lambda/(2*m) * (theta(2:b)')*theta(2:b);

% for J(1) we don't regularize
grad(1) = X(:,1)'*(1/m)*(h_theta' - y);

% Regularized gradient for J through number of features
grad(2:b) = X(:,(2:b))'*(1/m)*(h_theta' - y)  + lambda/m * theta(2:b);




% =============================================================

end
