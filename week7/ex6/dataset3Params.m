function [C, sigma] = dataset3Params(X, y, Xval, yval)

C = 1;
sigma = 0.03;

bestPrediction = 1000;
for C_i = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
  for sigma_i = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    model= svmTrain(X, y, C_i, @(x1, x2) gaussianKernel(x1, x2, sigma_i));
    predictions = svmPredict(model, Xval);
    prediction = mean(double(predictions ~= yval));
    if prediction < bestPrediction;
      bestPrediction = prediction;
      C = C_i;
      sigma = sigma_i;
    end
  end
end

end