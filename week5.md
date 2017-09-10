# Cost Function

Let's first define a few variables that we will need to use:

L = total number of layers in the network
sl = number of units (not counting bias unit) in layer l
K = number of output units/classes
Recall that in neural networks, we may have many output nodes. We denote hΘ(x)k as being a hypothesis that results in the kth output. Our cost function for neural networks is going to be a generalization of the one we used for logistic regression. Recall that the cost function for regularized logistic regression was:

`J(θ)=−1m∑mi=1[y(i) log(hθ(x(i)))+(1−y(i)) log(1−hθ(x(i)))]+λ2m∑nj=1θ2j`
For neural networks, it is going to be slightly more complicated:

`J(Θ)=−1m∑i=1m∑k=1K[y(i)klog((hΘ(x(i)))k)+(1−y(i)k)log(1−(hΘ(x(i)))k)]+λ2m∑l=1L−1∑i=1sl∑j=1sl+1(Θ(l)j,i)2`
We have added a few nested summations to account for our multiple output nodes. In the first part of the equation, before the square brackets, we have an additional nested summation that loops through the number of output nodes.

In the regularization part, after the square brackets, we must account for multiple theta matrices. The number of columns in our current theta matrix is equal to the number of nodes in our current layer (including the bias unit). The number of rows in our current theta matrix is equal to the number of nodes in the next layer (excluding the bias unit). As before with logistic regression, we square every term.

# Backpropagation Algorithm

"Backpropagation" is neural-network terminology for minimizing our cost function, just like what we were doing with gradient descent in logistic and linear regression. Our goal is to compute:

minΘJ(Θ)
That is, we want to minimize our cost function J using an optimal set of parameters in theta. In this section we'll look at the equations we use to compute the partial derivative of J(Θ):

∂∂Θ(l)i,jJ(Θ)
To do so, we use the following algorithm:


Back propagation Algorithm

Given training set {(x(1),y(1))⋯(x(m),y(m))}
Set Δ(l)i,j := 0 for all (l,i,j), (hence you end up having a matrix full of zeros)
For training example t =1 to m:

Set a(1):=x(t)
Perform forward propagation to compute a(l) for l=2,3,…,L

3. Using y(t), compute δ(L)=a(L)−y(t)
Where L is our total number of layers and a(L) is the vector of outputs of the activation units for the last layer. So our "error values" for the last layer are simply the differences of our actual results in the last layer and the correct outputs in y. To get the delta values of the layers before the last layer, we can use an equation that steps us back from right to left:

4. Compute δ(L−1),δ(L−2),…,δ(2) using δ(l)=((Θ(l))Tδ(l+1)) .∗ a(l) .∗ (1−a(l))
The delta values of layer l are calculated by multiplying the delta values in the next layer with the theta matrix of layer l. We then element-wise multiply that with a function called g', or g-prime, which is the derivative of the activation function g evaluated with the input values given by z(l).

The g-prime derivative terms can also be written out as:

g′(z(l))=a(l) .∗ (1−a(l))
5. Δ(l)i,j:=Δ(l)i,j+a(l)jδ(l+1)i or with vectorization, Δ(l):=Δ(l)+δ(l+1)(a(l))T
Hence we update our new Δ matrix.

D(l)i,j:=1m(Δ(l)i,j+λΘ(l)i,j), if j≠0.
D(l)i,j:=1mΔ(l)i,j If j=0
The capital-delta matrix D is used as an "accumulator" to add up our values as we go along and eventually compute our partial derivative. Thus we get ∂∂Θ(l)ijJ(Θ)= D(l)ij

# Backpropagation Intuition

Note: [4:39, the last term for the calculation for z31 (three-color handwritten formula) should be a22 instead of a21. 6:08 - the equation for cost(i) is incorrect. The first term is missing parentheses for the log() function, and the second term should be (1−y(i))log(1−hθ(x(i))). 8:50 - δ(4)=y−a(4) is incorrect and should be δ(4)=a(4)−y.]

Recall that the cost function for a neural network is:

J(Θ)=−1m∑t=1m∑k=1K[y(t)k log(hΘ(x(t)))k+(1−y(t)k) log(1−hΘ(x(t))k)]+λ2m∑l=1L−1∑i=1sl∑j=1sl+1(Θ(l)j,i)2
If we consider simple non-multiclass classification (k = 1) and disregard regularization, the cost is computed with:

cost(t)=y(t) log(hΘ(x(t)))+(1−y(t)) log(1−hΘ(x(t)))
Intuitively, δ(l)j is the "error" for a(l)j (unit j in layer l). More formally, the delta values are actually the derivative of the cost function:

δ(l)j=∂∂z(l)jcost(t)
Recall that our derivative is the slope of a line tangent to the cost function, so the steeper the slope the more incorrect we are. Let us consider the following neural network below and see how we could calculate some δ(l)j:


In the image above, to calculate δ(2)2, we multiply the weights Θ(2)12 and Θ(2)22 by their respective δ values found to the right of each edge. So we get δ(2)2= Θ(2)12*δ(3)1+Θ(2)22*δ(3)2. To calculate every single possible δ(l)j, we could start from the right of our diagram. We can think of our edges as our Θij. Going from right to left, to calculate the value of δ(l)j, you can just take the over all sum of each weight times the δ it is coming from. Hence, another example would be δ(3)2=Θ(3)12*δ(4)1.

# Implementation Note: Unrolling Parameters

With neural networks, we are working with sets of matrices:

Θ(1),Θ(2),Θ(3),…D(1),D(2),D(3),…
In order to use optimizing functions such as "fminunc()", we will want to "unroll" all the elements and put them into one long vector:

```
thetaVector = [ Theta1(:); Theta2(:); Theta3(:); ]
deltaVector = [ D1(:); D2(:); D3(:) ]
```

If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11, then we can get back our original matrices from the "unrolled" versions as follows:

```
Theta1 = reshape(thetaVector(1:110),10,11)
Theta2 = reshape(thetaVector(111:220),10,11)
Theta3 = reshape(thetaVector(221:231),1,11)
```

# Gradient Checking

Gradient checking will assure that our backpropagation works as intended. We can approximate the derivative of our cost function with:

`∂∂ΘJ(Θ)≈J(Θ+ϵ)−J(Θ−ϵ)2ϵ`

With multiple theta matrices, we can approximate the derivative with respect to Θj as follows:

`∂∂ΘjJ(Θ)≈J(Θ1,…,Θj+ϵ,…,Θn)−J(Θ1,…,Θj−ϵ,…,Θn)2ϵ`

A small value for ϵ (epsilon) such as ϵ=10−4, guarantees that the math works out properly. If the value for ϵ is too small, we can end up with numerical problems.

Hence, we are only adding or subtracting epsilon to the Θj matrix. In octave we can do it as follows:

```
epsilon = 1e-4;
for i = 1:n,
  thetaPlus = theta;
  thetaPlus(i) += epsilon;
  thetaMinus = theta;
  thetaMinus(i) -= epsilon;
  gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
end;
```

We previously saw how to calculate the deltaVector. So once we compute our gradApprox vector, we can check that gradApprox ≈ deltaVector.
Once you have verified once that your backpropagation algorithm is correct, you don't need to compute gradApprox again. The code to compute gradApprox can be very slow.

# Random Initialization

Initializing all theta weights to zero does not work with neural networks. When we backpropagate, all nodes will update to the same value repeatedly. Instead we can randomly initialize our weights for our Θ matrices using the following method:


Hence, we initialize each Θ(l)ij to a random value between[−ϵ,ϵ]. Using the above formula guarantees that we get the desired bound. The same procedure applies to all the Θ's. Below is some working code you could use to experiment.

If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.

```
Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
```

rand(x,y) is just a function in octave that will initialize a matrix of random real numbers between 0 and 1.

>Note: the epsilon used above is unrelated to the epsilon from Gradient Checking

# Putting it Together

First, pick a network architecture; choose the layout of your neural network, including how many hidden units in each layer and how many layers in total you want to have.
- Number of input units = dimension of features x(i)
- Number of hidden units per layer = usually more the better (must balance with cost of computation as it increases with more hidden units)
- Number of output units = number of classes
- Defaults: 1 hidden layer. If you have more than 1 hidden layer, then it is recommended that you have the same number of units in every hidden layer.

Training a Neural Network
1. Randomly initialize the weights
2. Implement forward propagation to get hΘ(x(i)) for any x(i)
3. Implement the cost function
4. Implement backpropagation to compute partial derivatives
5. Use gradient checking to confirm that your backpropagation works. Then disable gradient checking.
6. Use gradient descent or a built-in optimization function to minimize the cost function with the weights in theta.
7. When we perform forward and back propagation, we loop on every training example:

```
for i = 1:m,
   Perform forward propagation and backpropagation using example (x(i),y(i))
   (Get activations a(l) and delta terms d(l) for l = 2,...,L
```

The following image gives us an intuition of what is happening as we are implementing our neural network:
Ideally, you want hΘ(x(i)) ≈ y(i). This will minimize our cost function. However, keep in mind that J(Θ) is not convex and thus we can end up in a local minimum instead.
