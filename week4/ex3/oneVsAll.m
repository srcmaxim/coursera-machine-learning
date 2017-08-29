function [all_theta] = oneVsAll(X, y, num_labels, lambda)

m = size(X, 1);
n = size(X, 2);
X = [ones(m, 1) X];

initial_theta = zeros(n + 1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 50);

for class = 1:num_labels,
	[theta] = ...
		fmincg (@(t)(lrCostFunction(t, X, (y == class), lambda)), ...
			initial_theta, options);
	all_theta(class, :) = theta;
end

end
