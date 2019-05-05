function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    
    h = X * theta;
    
    err = (h-y);
    gradient1 = alpha * (1/m) * sum(err.*X(:,1));
    gradient2 = alpha * (1/m) * sum(err.*X(:,2));
    theta1 = theta(1)-gradient1;
    theta2 = theta(2)-gradient2;
    theta(1) = theta1;
    theta(2) = theta2;
    
    % ============================================================

end