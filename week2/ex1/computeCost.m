function J = computeCost(X, y, theta)

m = length(y);
J = 0;
X_m = size(X, 2);

for i=1:m
	XijQj = 0;
	for j=1:X_m
		XijQj = XijQj + X(i, j) * theta(j); 
	end
	J = J + (XijQj - y(i))^2;
end;

J = J/2/m;

end;
