function Z = projectData(X, U, K)

Z = zeros(size(X, 1), K);

Ureduce = U(:,1:K);
for i=1:size(X,1),
  Z(i,:) = X(i,:) * Ureduce;
end

end
