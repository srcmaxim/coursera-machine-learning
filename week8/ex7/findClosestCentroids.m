function idx = findClosestCentroids(X, centroids)

K = size(centroids, 1);
idx = zeros(size(X,1), 1);

for i=1:size(X,1) 
    for j = 1:K
        distance(j)= sum((centroids(j,:)-X(i,:)).^2);
    end
    %finds the indices of the minimum values
    [min_value, index] = min(distance);
    idx(i) = index; 
end

end

