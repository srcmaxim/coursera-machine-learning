function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

		t1 = theta(1) - alpha * mean(((X * theta) - y) .* X(:, 1));
		t2 = theta(2) - alpha * mean(((X * theta) - y) .* X(:, 2));

	theta(1) = t1;
	theta(2) = t2; 

    	J_history(iter) = computeCost(X, y, theta);

end

end
