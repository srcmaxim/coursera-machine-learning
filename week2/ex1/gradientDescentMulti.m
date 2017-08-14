function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

X
m = length(y); 
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

	hypothesis = X * theta;
	errors = hypothesis - y;
	decrement = alpha * (1/m) * errors' * X;
	theta = theta - decrement';
	J_history(iter) = computeCostMulti(X, y, theta);	

end;
end;
