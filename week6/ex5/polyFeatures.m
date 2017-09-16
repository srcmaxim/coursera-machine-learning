function [X_poly] = polyFeatures(X, p)
	
X_poly = zeros(numel(X), p);

for i=1:length(X),
    for j=1:p,
        X_poly(i, j) = X(i).^j;
    end
end

end
