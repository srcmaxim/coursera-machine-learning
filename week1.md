# Introduction

Used in:
- DB mining
- programming without human
- recomender services
- anderstanding human brain

Machine Learnig -- computers learn without being explicitly programmed.

ML algorithms:
- Supervised -- teach computer to do something. Dataset with right ansvers are given.
- Not supervised -- computer teaches by himself. Finds some structure of the data.
- Reinforsement -- output of an algorithm forming expirience for tuning algorithm.
- Recomender systems

Regression -> predict value output
Classification -> descrete output(0,1)

Supervised Example:
(a) Regression - Given a picture of a person, we have to predict their age on the basis of the given picture
(b) Classification - Given a patient with a tumor, we have to predict whether the tumor is malignant or benign.

Unsupervised Example:
(a) Clastering - Classification
(b) Non-clastering - Coctail party algorithm

Feature -- statement X, has some value

# Model and Cost Function

Training Set: (xi, yi) = (3, 5)

Training Set(x, y) --> Learning Algorithm --> Hypothesis (h)
x --> h --> y

h -- function that mappes x to y.
hq(x) = q0 + q1x -- linear regression with one variable.
qn -- parameters
When the target variable that were trying to predict is continuous -- regression problem. Small number of discrete values -- classification problem.

How to chose q0, q1?
Squared Error Cost Function: J(q0, q1) = 1 / (2m) \* E[i=1,m] (hq(xi) - yi)^2
To break it apart, it is 12 x where x is the mean of the squares of hq(xi)-yi , or the difference between the predicted value and the actual value.
On this data hq(xi)-yi hypotesis must be close to the real value.

This function is otherwise called the "Squared error function", or "Mean squared error". The mean is halved (12) as a convenience for the computation of the gradient descent, as the derivative term of the square function will cancel out the 12 term.

To visualize J(q0, q1) can be used 3d graph, but contour plotsmore convenient way.

# Parameter Learning

Gradient used to minimize function.

Qj:=Qj-a(d/(dQj))J(Q0,Q1)
a -- the lenth of the next step
d/(dQj) -- derivative for Q1 or Q2
Also updates of Q must be not implase but all Q changes in a time.

New Qj is a result of previous Qj and direction for J(Q0,Q1).
Direction for J(Q0,Q1) calculates as a(d/(dQj)J(Q0,Q1).

1. moving left right: derivative chouse where to go to as a A+Bx function.
If B >= 0 --> Qj:= Qj- value --> moveleft
If B <= 0 --> Qj:= Qj+ value --> moveright
2. lenth of step:
If a is low --> slop algorithm
If a is big --> exceptions in chousing
3. approaching:
If far from min --> [lim (d/(dQj) --> 1] --> steps bigger
If close to min --> [lim (d/(dQj) --> 0] --> steps smaller

Convex function -- 3d U-shape function

Gradient descent without iterative funcction.
Other is normal equation method --> Gradient descent scale better.
