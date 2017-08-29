function p = predict(Theta1, Theta2, X)
	
m = size(X, 1);
num_labels = size(Theta2, 1);

a1 = [ones(m, 1), X];

z2 = a1 * Theta1';
a2 = sigmoid(z2);
bias = ones(size(a2, 1), 1);
a2 = [bias, a2];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

[m, p] = max(a3, [], 2);

end
