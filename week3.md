# Classification

The classification problem is just like the regression problem, except that the values we now want to predict take on only a small number of discrete values. For now, we will focus on the binary classification problem in which y can take on only two values, 0 and 1. (Most of what we say here will also generalize to the multiple-class case.) For instance, if we are trying to build a spam classifier for email, then x(i) may be some features of a piece of email, and y may be 1 if it is a piece of spam mail, and 0 otherwise. Hence, y∈{0,1}. 0 is also called the negative class, and 1 the positive class, and they are sometimes also denoted by the symbols “-” and “+”.

We could approach the classification problem ignoring the fact that y is discrete-valued, and use our old linear regression algorithm to try to predict y given x. However, it is easy to construct examples where this method performs very poorly. Intuitively, it also doesn’t make sense for hθ(x) to take values larger than 1 or smaller than 0 when we know that y ∈ {0, 1}. To fix this, let’s change the form for our hypotheses hθ(x) to satisfy 0≤hθ(x)≤1. This is accomplished by plugging θ'x into the Logistic Function.

Our new form uses the __"Sigmoid Function"__, also called the "Logistic Function":

hθ(x)=g(θ'x)
z=θ'x
g(z)=1/(1+e^−z)

The function g(z), shown here, maps any real number to the (0, 1) interval, making it useful for transforming an arbitrary-valued function into a function better suited for classification.

hθ(x) will give us the probability that our output is 1. For example, hθ(x)=0.7 gives us a probability of 70% that our output is 1. Our probability that our prediction is 0 is just the complement of our probability that it is 1 (e.g. if probability that it is 1 is 70%, then the probability that it is 0 is 30%).

hθ(x)=P(y=1|x;θ)=1−P(y=0|x;θ)
P(y=0|x;θ)+P(y=1|x;θ)=1

In order to get our discrete 0 or 1 classification, we can translate the output of the hypothesis function as follows:

hθ(x)≥0.5→y=1
hθ(x)<0.5→y=0

The way our logistic function g behaves is that when its input is greater than or equal to zero, its output is greater than or equal to 0.5:

g(z)≥0.5 when z≥0

Remember.

>z=0,e^0=1⇒g(z)=1/2
z→∞,e−∞→0⇒g(z)=1
z→−∞,e∞→∞⇒g(z)=0

So if our input to g is θTX, then that means:

hθ(x)=g(θ'x)≥0.5 when θTx≥0
From these statements we can now say:

θ'x≥0⇒y=1
θ'x<0⇒y=0

The decision boundary is the line that separates the area where y = 0 and where y = 1. It is created by our hypothesis function.

# Cost Function

Simple mean of quadratic difference is __non-convex__ -- has local minimums. Hence we need to use another algorithm.

J(θ) = (1 / m) * ∑i=[1:m] Cost(hθ(x(i)), y(i))
Cost(hθ(x),y) = −log(hθ(x)) if y = 1
Cost(hθ(x),y) = −log(1−hθ(x))if y = 0

Cost(hθ(x),y) = 0 if hθ(x) = y
Cost(hθ(x),y) → ∞ if y=0 and hθ(x) → 1
Cost(hθ(x),y) → ∞ if y=1 and hθ(x) → 0

__Modified cost:__

Cost(hθ(x),y) = -ylog(hθ(x))- (1 - y)log(1 - hθ(x))

__Modified cost function:__

h=g(Xθ)
J(θ)=1/m⋅(−yTlog(h)−(1−y)Tlog(1−h))

__Gradient descent:__

Repeat{
  θj:=θj−α∂∂θjJ(θ)
}

θ:=θ−αmXT(g(Xθ)−y)

__Advanced optimization__

- Conjugate gradient
- BFGS
- L-BFGS

Cost function for min:

```octave
function [jVal, gradient] = costFunction(theta)
  jVal = [...code to compute J(theta)...];
  gradient = [...code to compute derivative of J(theta)...];
end
```

Evaluate:

```octave
options = optimset('GradObj', 'on', 'MaxIter', 100);
initialTheta = zeros(2,1);
   [optTheta, functionVal, exitFlag] = fminunc(@costFunction, initialTheta, options);
```

# Multiclass classification

__One-vs-rest:__

Now we will approach the classification of data when we have more than two categories. Instead of y = {0,1} we will expand our definition so that y = {0,1...n}.

Since y = {0,1...n}, we divide our problem into n+1 (+1 because the index starts at 0) binary classification problems; in each one, we predict the probability that 'y' is a member of one of our classes.

y∈{0,1...n}h(0)θ(x)=P(y=0|x;θ)h(1)θ(x)=P(y=1|x;θ)⋯h(n)θ(x)=P(y=n|x;θ)prediction=maxi(h(i)θ(x))



# Overfitting

Overfitting -- figure captured by a model but for new data algorithm don't work well.

1. Reduce the number of features:
  - Manually select which features to keep.
  - Use a model selection algorithm (studied later in the course).
2. Regularization
  - Keep all the features, but reduce the magnitude of parameters θj.
  - Regularization works well when we have a lot of slightly useful features.

#Regulation on linear regression

We'll want to eliminate the influence of θ3x3 and θ4x4 . Without actually getting rid of these features or changing the form of our hypothesis, we can instead modify our cost function:

minθ = (1/2m) * ∑ [i=1m] (hθ(x(i))−y(i))^2+1000⋅θ^2 3+1000⋅θ^2 4

We've added two extra terms at the end to inflate the cost of θ3 and θ4. Now, in order for the cost function to get close to zero, we will have to reduce the values of θ3 and θ4 to near zero. This will in turn greatly reduce the values of θ3x3 and θ4x4 in our hypothesis function. As a result, we see that the new hypothesis (depicted by the pink curve) looks like a quadratic function but fits the data better due to the extra small terms θ3x3 and θ4x4.

  minθ = (1/2m) ∑ [i=1m] (hθ(x(i))−y(i))^2+λ ∑ [j=1n] θ^2j

# Regulation gradient

Repeat {    
  θ0:=θ0−α 1m ∑i=1m(hθ(x(i))−y(i))x(i)0
  θj:=θj−α [(1m ∑ [i=1m] (hθ(x(i))−y(i))x(i)j)+λ/mθj]
}

# Normal equations

θ=(XTX+λ⋅L)−1XTy
L=eye(N) with 1,1=0

# Regulation logistic regression

J(θ) = −1/m ∑ [i=1m] [y(i) log(hθ(x(i)))+(1−y(i)) log(1−hθ(x(i)))]+λ/2m ∑ [j=1n] θ^2j






























.
