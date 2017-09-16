function [J, grad] = linearRegCostFunction(X, y, theta, lambda)

m = length(y)
J = 0;
grad = zeros(size(theta));

H = X * theta;
J = sum((H - y).^2)/(2*m);
J = J + sum(theta(2:end).^2)*lambda/(2*m);

gradientTheta = theta;
gradientTheta(1) = 0;

for i=1:length(theta),
    grad(i) = sum((H - y).*X(:,i))/m;
end;

grad = grad + gradientTheta*lambda/m;
grad = grad(:);

end
