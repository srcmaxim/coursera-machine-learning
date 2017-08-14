function J = computeCostMulti(X, y, theta)

	J = 1/2 * mean((X*theta-y).^2);

end;

