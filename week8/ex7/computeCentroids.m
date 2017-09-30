function centroids = computeCentroids(X, idx, K)

[m n] = size(X);
centroids = zeros(K, n);

for c = 1:K,
  count = 0;
  for i = 1:m,
    	if idx(i) == c
      	centroids(c,:) = centroids(c,:) + X(i,:);
      	count = count + 1;
    	end
  end  
	if count > 0
    		centroids(c,:) = centroids(c,:) ./ count;
  end
end
end

