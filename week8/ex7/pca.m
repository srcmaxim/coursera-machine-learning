function [U, S] = pca(X)

[m, n] = size(X);

% You need to return the following variables correctly.
U = zeros(n);
S = zeros(n);

sigma = X' * X ./ m;
[U, S, V] = svd(sigma);

end
