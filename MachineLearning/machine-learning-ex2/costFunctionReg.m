function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
h = X*theta;
firstY = y.*(-1);
secY = 1.-y;
first = (1/m)*(firstY.*log(sigmoid(h)));
second = (1/m)*(secY.*log(1-sigmoid(h)));
reg = (lambda/(2*m))*(theta.^2);
costFunc = first-second;
J1 = sum(costFunc);
J2 = sum(reg)- reg(1);
J = J1+J2;

grad = (1/m)*(X'*(sigmoid(h)-y));
tempGrad = grad(1);
grad = (1/m)*(X'*(sigmoid(h)-y))+(lambda/m)*theta;
grad(1) = tempGrad;
% =============================================================

end
