function X_rec = recoverData(Z, U, K)

X_rec = zeros(size(Z, 1), size(U, 1));

for i=1:size(Z, 1),
  X_rec(i,:) = Z(i,:) * U(:, 1:K)';
end

end
