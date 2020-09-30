import nn
#import time

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    # Q1
    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.get_weights(),x)


    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        dot_product = nn.as_scalar(self.run(x))
        if dot_product >= 0:
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        #nn.Parameter.update()
        misclassified = True
        while misclassified:
            misclassified = False
            for x, y in dataset.iterate_once(1):  #??
                #print(x)
                #time.sleep(10)
                if self.get_prediction(x) != nn.as_scalar(y):
                    misclassified = True
                    self.w.update(x, nn.as_scalar(y))

    ## end Q1 -  python autograder.py -q q1

# linearModel()
# m = nn.Parameter(2, 1)
# b = nn.Parameter(1, 1)
#
# xm = nn.Linear(x, m)
# predicted_y = nn.AddBias(xm, b)
#
# loss = nn.SquareLoss(predicted_y, y)
#
# grad_wrt_m, grad_wrt_b = nn.gradients(loss, [m, b])
# multiplier = 0.1 #
# m.update(grad_wrt_m, multiplier)
#  include an update for b and add a loop to repeatedly perform gradient updates,
#  we will have the full training procedure for linear regression.

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 10
        self.w1 = nn.Parameter(1, 20)
        self.w2 = nn.Parameter(20, 10)
        self.w3 = nn.Parameter(10, 1)

        self.b1 = nn.Parameter(1, 20)
        self.b2 = nn.Parameter(1, 10)
        self.b3 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        r1 = nn.AddBias(nn.Linear(x, self.w1), self.b1)
        r2 = nn.AddBias(nn.Linear(nn.ReLU(r1), self.w2), self.b2)
        y = nn.AddBias(nn.Linear(nn.ReLU(r2), self.w3), self.b3)
        return y

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        epsilon = 0.005
        base_rate = -0.05 #/ self.batch_size/10     #dynamic learning rate
        dynamic_rate = -0.05 #/ self.batch_size/10
        n = 0
        misclassified = True
        while misclassified:
            misclassified = False
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                if nn.as_scalar(loss) > epsilon:
                    misclassified = True
                    n += 1
                    learning_rate = base_rate + dynamic_rate*(1/(1+0.001*n))
                    gradients_list = nn.gradients(loss,
                                                  [self.w1, self.w2, self.w3, self.b1, self.b2, self.b3])
                    self.w1.update(gradients_list[0], learning_rate)
                    self.w2.update(gradients_list[1], learning_rate)
                    self.w3.update(gradients_list[2], learning_rate)
                    self.b1.update(gradients_list[3], learning_rate)
                    self.b2.update(gradients_list[4], learning_rate)
                    self.b3.update(gradients_list[5], learning_rate)
        print(n)

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 60

        self.w1 = nn.Parameter(784, 300)
        self.w2 = nn.Parameter(300, 100)
        self.w3 = nn.Parameter(100, 10)

        self.b1 = nn.Parameter(1, 300)
        self.b2 = nn.Parameter(1, 100)
        self.b3 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        r1 = nn.AddBias(nn.Linear(x, self.w1), self.b1)
        r2 = nn.AddBias(nn.Linear(nn.ReLU(r1), self.w2), self.b2)
        y = nn.AddBias(nn.Linear(nn.ReLU(r2), self.w3), self.b3)
        return y


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        #epsilon = 0.03
        base_rate = -0.05  # / self.batch_size/10     #dynamic learning rate
        dynamic_rate = -0.5  # / self.batch_size/10
        n = 0; c = 0;
        misclassified = True
        while misclassified:
            misclassified = False
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                #acc = dataset.get_validation_accuracy()
                #print(n )#"  acc:  " ,acc)#"   loss: ", nn.as_scalar(loss))
                if  c < 500 :  #acc < 0.98:     #nn.as_scalar(loss) > epsilon:
                    misclassified = True
                    n += 1; c +=1;
                    learning_rate = base_rate + dynamic_rate * (1 / (1 + 0.0005 * n))
                    gradients_list = nn.gradients(loss, [self.w1, self.w2, self.w3, self.b1, self.b2, self.b3])
                    self.w1.update(gradients_list[0], learning_rate)
                    self.w2.update(gradients_list[1], learning_rate)
                    self.w3.update(gradients_list[2], learning_rate)
                    self.b1.update(gradients_list[3], learning_rate)
                    self.b2.update(gradients_list[4], learning_rate)
                    self.b3.update(gradients_list[5], learning_rate)
                else:
                    acc = dataset.get_validation_accuracy()
                    if acc < 0.98:  #977
                        print(n, "  acc: ", acc)
                        misclassified = True
                        c = 0
                    else:
                        break
        print("end at: ", n, "   acc: ",dataset.get_validation_accuracy())
        #print(dataset.get_validation_accuracy())

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 50
        self.neurons = 400

        self.w = nn.Parameter(self.num_chars, self.neurons)    #47*9=423   *12=564
        self.h = nn.Parameter(self.neurons, self.neurons)
        self.hf = nn.Parameter(self.neurons, 5)

        self.b = nn.Parameter(1, self.neurons)
        self.bh = nn.Parameter(1, self.neurons)
        self.bf = nn.Parameter(1, 5)

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        z = nn.AddBias(nn.Linear(xs[0], self.w), self.b)
        #z = nn.AddBias(nn.Linear(nn.ReLU(z), self.h2), self.b2)
        z = nn.ReLU(z)

        for x in xs[1:]:
            z0 = nn.AddBias(nn.Linear(x, self.w), self.b)
            z1 = nn.AddBias(nn.Linear(z, self.h), self.bh)
            z = nn.Add(z0, z1)
            z = nn.ReLU(z)
        return nn.AddBias(nn.Linear(z, self.hf), self.bf)


    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        base_rate = -0.01  # / self.batch_size/10     #dynamic learning rate
        dynamic_rate = -0.2  # / self.batch_size/10
        n = 0; c = 0
        misclassified = True
        while misclassified:
            misclassified = False
            for x, y in dataset.iterate_once(self.batch_size):
                loss = self.get_loss(x, y)
                # acc = dataset.get_validation_accuracy()
                #print(n , end=',')#"  acc:  " ,acc)#"   loss: ", nn.as_scalar(loss))
                if c < 500:  # acc < 0.98:     #nn.as_scalar(loss) > epsilon:
                    misclassified = True
                    n += 1; c += 1
                    learning_rate = base_rate + dynamic_rate * (1 / (1 + 0.0005 * n))
                    gradients_list = nn.gradients(loss, [self.w, self.h, self.hf, self.b, self.bh, self.bf])
                    self.w.update(gradients_list[0], learning_rate)
                    self.h.update(gradients_list[1], learning_rate)
                    self.hf.update(gradients_list[2], learning_rate)
                    self.b.update(gradients_list[3], learning_rate)
                    self.bh.update(gradients_list[4], learning_rate)
                    self.bf.update(gradients_list[5], learning_rate)
                else:
                    acc = dataset.get_validation_accuracy()
                    if acc < 0.89:
                        print(n, " acc: ", acc)
                        misclassified = True
                        c = 0
                    else:
                        break
        print("end at: ", n, "   acc: ", dataset.get_validation_accuracy())
