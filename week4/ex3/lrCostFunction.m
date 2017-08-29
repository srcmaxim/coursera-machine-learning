function [J, grad] = lrCostFunction(theta, X, y, lambda)

m = length(y);
h = sigmoid(X * theta);
cost = (1 / m) * sum((-y .* log(h)) - ((1 - y) .* log(1 - (h))));
regCost = (lambda / (2 * m)) * norm(theta([2:end])) ^ 2;
grad = (1 / m) .* X' * (h - y);
regGrad = (lambda / m) .* theta;
regGrad(1) = 0;
J = cost + regCost;
grad = grad + regGrad;
grad = grad(:);

end
