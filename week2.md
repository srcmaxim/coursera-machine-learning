# Multiple Linear Regression

Linear regression with multiple variables is also known as "multivariate linear regression".

We now introduce notation for equations where we can have any number of input variables.

x^(i)j = value of the freature
x^(i) = input (row) of concrete features
m = number of trainig exampples
n = number of features

The __multivariable hypothesis__ function accommodating these multiple features is as follows:

hθ(x) = θ0 + θ1X1 + θ2X2 + θ3X3 +...+ θnXn

Q = [Q0 Q1 Q2]
X = [1 2 3; 2 3 4; 3 4 5]
hQ(X) = X*Q'

This is a vectorization of our hypothesis function for one training example; see the lessons on vectorization to learn more.
How elements are described: x(i)0=1 for (i∈1,…,m).

__Vectorization of hypothesis evaluation__:
X*Q'-Y

__Vectorization of gradient descent__:
Qj := Qj - a ⋅ (d / (dQj)) ⋅ J(Q)
Qj := Qj - a ⋅ (d / (dQj)) ⋅ 1 / (2m) ⋅ E[i=1,m] (hq(xi) - yi)^2
Qj := Qj - a / m ⋅ E[i=1,m] (hq(xi) - yi) ⋅ Xij

Xij = j element of X array

repeat until convergence:{
  θj:=θj−α1m∑i=1m(hθ(x(i))−y(i))⋅x(i)j        for j := 0...n
}

Q = Q - a * mean((X*Q'-Y), 2) * deriviate(X*Q' by Q)

__Feature Scaling__ -- gradient descent will run faster
Xi = (Xi - avg(x)) / (max(x) - min(x))

__Tresholds__

-3...3
10^-3...10^3

__Different Function Regressions__

hθ(x) = θ0 + θ1*x1 + θ2*x1^2 + θ3*x1^3
hθ(x) = θ0 + θ1*x1 + θ2*sqrt(x1)

__Normal Equation___

Q = (X'X)^-1X'Y -- not good for big N, no need chose a

If X is not invertable?
1. delete equal features
2. to many features

# Octave Tutorial

__math operations__
PS1('>> ')
1 == 2
3 ~= 5
4 - 2
4 * 8
1 && 0
1 || 0
xor(1, 0)

__assigment__
a = 3
a = 3; -- semicolon suppressing output
b = 'hi'
a -- prints
disp(printf('2 decimals: %0.2d', pi))
format long, formal short -- how many display values to display


__matrix__
A = [1 2; 3 4; 5 6] = [1, 2; 3, 4; 5, 6] -- 3 by 2 matrix
B = [1; 2; 3] -- vector 3 by 1
A' -- transpose

1:0.1:2 -- range matrix with 0.1 step
1:10 -- range matrix with 1 step

ones(3, 3)
zeros(3, 3)
eye(4) -- 4 by 4 identity matrix

rand(3, 3) -- random matrix
randn(3, 3) -- random gaussian matrix

size(A)
size(A, 1) -- size of first dimention
length(V) -- size of the longest dimention

__process data__
hist(A, 50) -- display frequency, 50 is for scale
help <function>
clc -- clear
ctrl + l -- clear

load featuresX.dat
load('pricesY.dat')
save hello.mat v -- save variable to a file
save hello.dat v -ascii --- save as text

cd
pwd
dir/ls

who -- display vars
whos -- display vars and their memory taken
clear -- deletes all vars

__manipulate data__
A = [1 2; 3 4; 5 6]
A(3, 2) => 6
A(:, 1) => [1; 3; 5]
A(2, :) => [3 4]
A(2:3, 1:2) => slice matrix [3 4; 5 6]
A(2, :) = [3, 4] -- make assignment
M = [A; B] -- concatenate vector A on top of B
M = [A, B] -- appends vector A to the left of B
A(:) -- put all elements of A in a single vector

__computing on data__
A*B -- multipy matrix
A+B-- add all elements by all
A.\*B -- multiply all elements by all
A.^2
A./2
log(v)
abs(v)
-v
v+1
max = max(v) -- max elements
[max, maxIDX] = max(v)
v < 3 -- returns element vice comparison
find(v < 4) -- witch elements IDX
magic(3) -- matrix rows cols returns the same number when multiply
[row, col] = find(magic(3) < 7)
sum(v)
prod(v)
floor(v), ceil(v), round(v)
max(rand(3), rand(3)) -- max element of matrixes
max(A, [], 1) -- max on 1st dimension of a
max(max(A)) -- max of matrix
max(A(:)) -- max of matrix
sum(A, 2) -- sums each row of a
flipup(A) -- flips matrix verticaly
pinv(A) -- inverts matrix

__plot results__

t = [0:0.01:0.99];
y1 = sin(2*pi*4*t);
plot(t, y1);
hold on; -- plot figure on top
y2 = cos(2*pi*4*t);
plot(t, y2);

xlabel('time')
ylabel('value')
legend('sin', 'cos')
print -dpng 'plot.png'
close

figure(1); plot(t, y1);
figure(2); plot(t, y2);
subplot(1, 2, 1) -- divide plot 1 x 2 and use 1 plot
plot(t, y1);
subplot(1, 2, 2) -- divide plot 1 x 2 and use 1 plot
plot(t, y2);
axis([0.5 1 -1 1]) -- change scale
clf -- clears all plot

imagesc(A) -- display matrix grid of colors
imagesc(A), colorbar, colormap gray;

__logic__

v = 1:10
for i = 1:10
  v(i) = i^2;
end;

i = 1;
while i <= 5
  v(i) = 100;
  i = i + 1;
end;

if i < 3
  do();
end;

__functions__

To create a maxMin function create file maxMin.m:
Openup Wordpad and type:
```
function [max, min] = maxMin(A)
  max = max(A);
  min = min(A);
end
```
go to directory or
addpath('D:\\ML')
Call function: [a, b] = maxMin(A)

```
function J = costFunctionJ(X, y, theta)
    m = size(X, 1);
    predictions = X*theta;
    sqrErrors = (predictions-y).^2'
    J = 1 / (2 * m) * sum(sqrErrors);
end;
```

__vectoruzation__

hypothesis:
hq(x) = E[j=0,n]Qj*xj
      = Q'x

gradient descent:
Q = Q - a \* d
d = 1 / m \* E[i=0,m] (hq(x^(i)-y^(i)))\*x^(i)
  = mean(Q' \* x - y) \* x

# Functions

```
function [theta] = normalEqn(X, y)

theta = pinv(X'*X)*X'*y;

end
```

```
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
```

```
function J = computeCostMulti(X, y, theta)

	J = 1/2 * mean((X*theta-y).^2);

end;
```

```
function [X_norm, mu, sigma] = featureNormalize(X)

mu = mean(X);
sigma = std(X);

% bsxfun applies function element-by-element to two maticies
X_norm = bsxfun(@minus, X, mu);
X_norm = bsxfun(@rdivide, X_norm, sigma);

end;
```

```
function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)

m = length(y); % number of training examples;
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

		t1 = theta(1) - alpha * mean(((X * theta) - y) .* X(:, 1));
		t2 = theta(2) - alpha * mean(((X * theta) - y) .* X(:, 2));

	theta(1) = t1;
	theta(2) = t2;

    	J_history(iter) = computeCost(X, y, theta);

end;
end;
```

```
function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)


m = length(y);
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

	hypothesis = X * theta;
	errors = hypothesis - y;
	decrement = alpha * (1/m) * errors' * X;
	theta = theta - decrement';
	J_history(iter) = computeCostMulti(X, y, theta);

end;
end;
```























.
