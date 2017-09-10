function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

%% Part 1 implementation

a1 = [ones(m, 1) X];

z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1), 1) a2];

z3 = a2 * Theta2';
a3 = sigmoid(z3);
hThetaX = a3;

yVec = zeros(m,num_labels);

for i = 1:m
    yVec(i,y(i)) = 1;
end

% for i = 1:m
%
%     term1 = -yVec(i,:) .* log(hThetaX(i,:));
%     term2 = (ones(1,num_labels) - yVec(i,:)) .* log(ones(1,num_labels) - hThetaX(i,:));
%     J = J + sum(term1 - term2);
%
% end
%
% J = J / m;

J = 1/m * sum(sum(-1 * yVec .* log(hThetaX)-(1-yVec) .* log(1-hThetaX)));

regularator = (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) * (lambda/(2*m));

J = J + regularator;

%% Part 2 implementation

for t = 1:m

	% For the input layer, where l=1:
	a1 = [1; X(t,:)'];

	% For the hidden layers, where l=2:
	z2 = Theta1 * a1;
	a2 = [1; sigmoid(z2)];

	z3 = Theta2 * a2;
	a3 = sigmoid(z3);

	yy = ([1:num_labels]==y(t))';
	% For the delta values:
	delta_3 = a3 - yy;

	delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(z2)];
	delta_2 = delta_2(2:end); % Taking of the bias row

	% delta_1 is not calculated because we do not associate error with the input

	% Big delta update
	Theta1_grad = Theta1_grad + delta_2 * a1';
	Theta2_grad = Theta2_grad + delta_3 * a2';
end

Theta1_grad = (1/m) * Theta1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
Theta2_grad = (1/m) * Theta2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
