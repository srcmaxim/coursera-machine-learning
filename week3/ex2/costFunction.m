function [J, grad] = costFunction(theta, X, y)

m = length(y);
h = sigmoid(X * theta);
J = ((-y)'*log(h) - (1-y)'*log(1-h))/m;
grad = ((h - y)'*X)/m;

end
