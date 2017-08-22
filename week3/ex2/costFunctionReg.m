function [J, grad] = costFunctionReg(theta, X, y, lambda)

m = length(y);
h = sigmoid(X*theta);
theta1 = [0 ; theta(2:size(theta), :)];
p = lambda*(theta1'*theta1)/(2*m);
J = ((-y)'*log(h) - (1-y)'*log(1-h))/m + p;
grad = (X'*(h - y)+lambda*theta1)/m;

end
