function [X_norm, mu, sigma] = featureNormalize(X)

mu = mean(X);
sigma = std(X);

% bsxfun applies function element-by-element to two maticies
X_norm = bsxfun(@minus, X, mu);
X_norm = bsxfun(@rdivide, X_norm, sigma);

end;
