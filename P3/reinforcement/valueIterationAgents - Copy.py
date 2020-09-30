# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    ### 1
    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            newStatesValue = self.values.copy()

            for state in self.mdp.getStates():
                QValuesList = []

                if not self.mdp.isTerminal(state):

                    for action in self.mdp.getPossibleActions(state):
                        QValuesList.append(self.computeQValueFromValues(state, action))

                    newStatesValue[state] = max(QValuesList)
            self.values = newStatesValue

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    ### 1
    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()

        QValue = 0.
        for nextState, probability in self.mdp.getTransitionStatesAndProbs(state, action):
            QValue += probability * (
                    self.mdp.getReward(state, action, nextState)
                    + self.discount * self.values[nextState])
        return QValue

    ### 1
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state          given self.value
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        if self.mdp.isTerminal(state):
            return None

        possibleActions = []
        QValues = []

        for action in self.mdp.getPossibleActions(state):
            possibleActions.append(action)
            QValues.append(self.getQValue(state, action) )

        return possibleActions[QValues.index(max(QValues))]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

### 4
class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    ### 4
    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        possibleStates = self.mdp.getStates()
        for i in range(self.iterations):
            state = possibleStates[i % len(possibleStates)]
            if not self.mdp.isTerminal(state):
                QValues = []
                for action in self.mdp.getPossibleActions(state):
                    QValues.append(self.getQValue(state, action) )
                self.values[state] = max(QValues)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    ### 5
    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        possibleStates = self.mdp.getStates()
        # initialize empty priority queue
        updateQueue = util.PriorityQueue()

        # Compute predecessors of all states
        predecessors = dict()
        for state in possibleStates:
            predecessors[state] = set()

        for s in possibleStates:
            #QValues = util.Counter()
            QValues = []
            for action in self.mdp.getPossibleActions(s):
                for (nextState, probability) in self.mdp.getTransitionStatesAndProbs(s, action):
                    if probability != 0:
                        predecessors[nextState].add(s)
                #QValues[action] = self.computeQValueFromValues(s, action)
                QValues.append(self.computeQValueFromValues(s, action)  )

            # for each non-terminal state s do
            # Find the absolute value of the difference ; call this number diff.
            # Do NOT update self.values[s] in this step.
            if not self.mdp.isTerminal(s):
                #Qmax = QValues[QValues.argMax()]
                #diff = abs(self.values[s] - Qmax)
                diff = abs(self.values[s] - max(QValues))
                updateQueue.update(s, -diff)

        # for iteration in 0, 1, 2, ..., self.iterations - 1, do
        # for i in xrange(self.iterations):
        i = 0
        while i < self.iterations:
            # If the priority queue is empty, then terminate.
            if updateQueue.isEmpty():
                return

            s = updateQueue.pop()
            if not self.mdp.isTerminal(s):
                #QValues = util.Counter()  #list
                QValues = []
                for action in self.mdp.getPossibleActions(s):
                    #QValues[action] = self.computeQValueFromValues(s, action)
                    QValues.append(self.computeQValueFromValues(s, action)   )
                #self.values[s] = QValues[QValues.argMax()]
                self.values[s] = max(QValues)
                # for each predecessor p of s, do
                for p in predecessors.get(s):
                    #QValues = util.Counter()
                    QValues = []
                    for action in self.mdp.getPossibleActions(p):
                        #QValues[a] = self.computeQValueFromValues(p, action)
                        QValues.append(self.computeQValueFromValues(p, action)  )
                    # qpMax = QValues[QValues.argMax()]
                    # diff = abs(self.values[p] - qpMax)
                    diff = abs(self.values[p] - max(QValues))

                    if diff > self.theta:
                        updateQueue.update(p, -diff)

            i += 1
