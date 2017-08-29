# Model Representation I

Let's examine how we will represent a hypothesis function using neural networks. At a very simple level, neurons are basically computational units that take inputs (dendrites) as electrical inputs (called "spikes") that are channeled to outputs (axons). In our model, our dendrites are like the input features x1⋯xn, and the output is the result of our hypothesis function. In this model our x0 input node is sometimes called the "bias unit." It is always equal to 1. In neural networks, we use the same logistic function as in classification, 1 / (1 + e ^ (- Q'X)), yet we sometimes call it a sigmoid (logistic) activation function. In this situation, our "theta" parameters are sometimes called "weights".

Visually, a simplistic representation looks like:

[x0; x1; x2] → [   ] → hθ(x)

Our input nodes (layer 1), also known as the "input layer", go into another node (layer 2), which finally outputs the hypothesis function, known as the "output layer".

We can have intermediate layers of nodes between the input and output layers called the "hidden layers."

In this example, we label these intermediate or "hidden" layer nodes a20⋯a2n and call them "activation units."

a(j)i="activation" of unit i in layer
jΘ(j)=matrix of weights controlling function mapping from layer j to layer j+1

If we had one hidden layer, it would look like:

[x0; x1; x2; x3] → [a(2)1; a(2)2; a(2)3] → hθ(x)

The values for each of the "activation" nodes is obtained as follows:

a(2)1 = g(Θ(1)10 * x0 + Θ(1)11 * x1 + Θ(1)12 * x2 + Θ(1)13 * x3)
a(2)2 = g(Θ(1)20 * x0 + Θ(1)21 * x1 + Θ(1)22 * x2 + Θ(1)23 * x3)
a(2)3 = g(Θ(1)30 * x0 + Θ(1)31 * x1 + Θ(1)32 * x2 + Θ(1)33 * x3)
hΘ(x) = a(3)1
= g(Θ(2)10 * a(2)0 + Θ(2)11 * a(2)1 + Θ(2)12 * a(2)2 + Θ(2)13 * a(2)3)

This is saying that we compute our activation nodes by using a 3×4 matrix of parameters. We apply each row of the parameters to our inputs to obtain the value for one activation node. Our hypothesis output is the logistic function applied to the sum of the values of our activation nodes, which have been multiplied by yet another parameter matrix Θ(2) containing the weights for our second layer of nodes.

Each layer gets its own matrix of weights, Θ(j).

The dimensions of these matrices of weights is determined as follows:

If network has sj units in layer j and sj+1 units in layer j+1, then Θ(j) will be of dimension sj+1×(sj+1).
The +1 comes from the addition in Θ(j) of the "bias nodes," x0 and Θ(j)0. In other words the output nodes will not include the bias nodes while the inputs will. The following image summarizes our model representation:


Example: If layer 1 has 2 input nodes and layer 2 has 4 activation nodes. Dimension of Θ(1) is going to be 4×3 where sj=2 and sj+1=4, so sj+1×(sj+1)=4×3.

# Model Representation II

To re-iterate, the following is an example of a neural network:

a(2)1 = g(Θ(1)10 * x0 + Θ(1)11 * x1 + Θ(1)12 * x2 + Θ(1)13 * x3)
a(2)2 = g(Θ(1)20 * x0 + Θ(1)21 * x1 + Θ(1)22 * x2 + Θ(1)23 * x3)
a(2)3 = g(Θ(1)30 * x0 + Θ(1)31 * x1 + Θ(1)32 * x2 + Θ(1)33 * x3)
hΘ(x) = a(3)1
= g(Θ(2)10 * a(2)0 + Θ(2)11 * a(2)1 + Θ(2)12 * a(2)2 + Θ(2)13 * a(2)3)

In this section we'll do a vectorized implementation of the above functions. We're going to define a new variable z(j)k that encompasses the parameters inside our g function. In our previous example if we replaced by the variable z for all the parameters we would get:

a(2)1 = g(z(2)1)
a(2)2 = g(z(2)2)
a(2)3 = g(z(2)3)

In other words, for layer j=2 and node k, the variable z will be:

z(2)k = Θ(1)k,0 * x0 + Θ(1)k,1 * x1 +⋯+ Θ(1)k,n * xn

The vector representation of x and zj is:

x = [x0x1⋯xn]
z(j) = [z(j)1; z(j)2; ⋯ z(j)n]

Setting x = a(1), we can rewrite the equation as:

z(j) = Θ(j−1) * a(j−1)

We are multiplying our matrix Θ(j−1) with dimensions sj×(n+1) (where sj is the number of our activation nodes) by our vector a(j−1) with height (n+1). This gives us our vector z(j) with height sj. Now we can get a vector of our activation nodes for layer j as follows:

a(j) = g(z(j))
Where our function g can be applied element-wise to our vector z(j).

We can then add a bias unit (equal to 1) to layer j after we have computed a(j). This will be element a(j)0 and will be equal to 1. To compute our final hypothesis, let's first compute another z vector:

z(j+1) = Θ(j) * a(j)

We get this final z vector by multiplying the next theta matrix after Θ(j−1) with the values of all the activation nodes we just got. This last theta matrix Θ(j) will have only one row which is multiplied by one column a(j) so that our result is a single number. We then get our final result with:

hΘ(x) = a(j+1) = sg(z(j+1))

Notice that in this last step, between layer j and layer j+1, we are doing exactly the same thing as we did in logistic regression. Adding all these intermediate layers in neural networks allows us to more elegantly produce interesting and more complex non-linear hypotheses.

# Examples and Intuitions I

A simple example of applying neural networks is by predicting x1 AND x2, which is the logical 'and' operator and is only true if both x1 and x2 are 1.

The graph of our functions will look like:

[x0; x1; x2] → [g(z(2))] → hΘ(x)

Remember that x0 is our bias variable and is always 1.

Let's set our first theta matrix as:

Θ(1) = [−30, 20, 20]

This will cause the output of our hypothesis to only be positive if both x1 and x2 are 1. In other words:

hΘ(x) = g(−30+20x1+20x2)

x1=0  and  x2=0  then  g(−30)≈0
x1=0  and  x2=1  then  g(−10)≈0
x1=1  and  x2=0  then  g(−10)≈0
x1=1  and  x2=1  then  g(10)≈1

# Examples and Intuitions II

The Θ(1) matrices for AND, NOR, and OR are:

AND:Θ(1) = [−30, 20, 20]
NOR:Θ(1) = [10, −20, −20]
OR:Θ(1) = [−10, 20, 20]

We can combine these to get the XNOR logical operator (which gives 1 if x1 and x2 are both 0 or both 1).

[x0, x1, x2] → [a(2)1, a(2)2] → [a(3)] → hΘ(x)

For the transition between the first and second layer, we'll use a Θ(1) matrix that combines the values for AND and NOR:

Θ(1)=[−30 10 20; −20 20 −20]

For the transition between the second and third layer, we'll use a Θ(2) matrix that uses the value for OR:

Θ(2)=[−10 20 20]

Let's write out the values for all our nodes:

a(2) = g(Θ(1) * x)
a(3) = g(Θ(2) * a(2))
hΘ(x) = a(3)
