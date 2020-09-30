# multiAgents.py
# --------------
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

import time
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)

        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        #print('max score:', bestScore,'  index:',chosenIndex)
        # index = 1 not move

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # python autograder.py
        # python autograder.py -q q2   #no graphic
        # python autograder.py -t test_cases/q2/0-small-tree

        # python pacman.py -p ReflexAgent
        # python pacman.py -p ReflexAgent -l testClassic
        # python pacman.py --frameTime 0 -p ReflexAgent -k 1

        # python autograder.py -q q1
        # python autograder.py -q q1 --no-graphics
        currentFoodList = newFood.asList()
        # get closes food:
        min_food_dis = 9999
        #min_food_count = 0
        score = 0. #+ random.random()
        #print(action)
        if action == Directions.STOP:
            #print("S")
            score -= 5
        # if action == Directions.WEST:
        #     print("W")
        #     score += 0.01
        # if action == Directions.NORTH:
        #     print("N")
        #     score += 0.01
        for i in currentFoodList:
            dis = manhattanDistance(newPos, i)
            if min_food_dis >= dis:
                min_food_dis = dis
        #print(min_food_dis)
        score += 10./min_food_dis
        #print(newGhostStates[0])


        for ghost_state in successorGameState.getGhostPositions():
            g_dis = manhattanDistance(newPos, ghost_state)
            if g_dis <= 1:
                score -= 9999
            if g_dis != 0:
                score -= 1./g_dis
        #/(len(newGhostStates)+1)
        #print('ghost dis', total_ghost_dis)
        #print(score)
        #print(len(newGhostStates))
        return score + successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

###################################################
#         Minimax Agent                         #
###################################################

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        # python autograder.py -q q2
        # python autograder.py -q q2 --no-graphics

        def max_value(nextAgent, agent, depth, gameState):
            v = float("-inf")
            for newState in gameState.getLegalActions(agent):
                v = max(v, value(nextAgent, depth, gameState.generateSuccessor(agent, newState)))
            return v
            #return max(value(1, depth, gameState.generateSuccessor(agent, newState)) for newState in
            #           gameState.getLegalActions(agent))

        def min_value(nextAgent, agent, depth, gameState):
            v = float("inf")
            for newState in gameState.getLegalActions(agent):
                v = min(v, value(nextAgent, depth, gameState.generateSuccessor(agent, newState)))
            return v
            #return min(value(agent, depth, gameState.generateSuccessor(agent, newState)) for newState in
            #           gameState.getLegalActions(agent))

        def value(agent, depth, gameState):
            # terminal state, return state's utility
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)
            # agent = pacman, max
            if agent == 0:
                return max_value(1, agent, depth, gameState)
            #max(value(1, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))
            # agent != pacman; min
            else:
                # 0 pacman; 1; 2; 3;
                nextAgent = agent + 1
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0  #if nextAgent == 0:
                    depth += 1
                return min_value(nextAgent, agent, depth, gameState)
                #min(value(nextAgent, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))


        maximum = float("-inf")
        #action = Directions.WEST
        for agentState in gameState.getLegalActions(0):
            utility = value(1, 0, gameState.generateSuccessor(0, agentState))
            if utility > maximum or maximum == float("-inf"):
                maximum = utility
                action = agentState

        return action


###################################################
#         Alpha Beta Agent                        #
###################################################

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        # python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
        # python autograder.py -q q3
        # python autograder.py -q q3 --no-graphics

        def ab_max_value(nextAgent, agent, depth, gameState, a, b):
            v = float("-inf")
            for newState in gameState.getLegalActions(agent):
                v = max(v, alpha_beta(nextAgent, depth, gameState.generateSuccessor(agent, newState), a, b))
                if v > b:
                    return v
                a = max(a, v)
            return v


        def ab_min_value(nextAgent, agent, depth, gameState, a, b):
            v = float("inf")
            for newState in gameState.getLegalActions(agent):
                v = min(v, alpha_beta(nextAgent, depth, gameState.generateSuccessor(agent, newState),  a, b))
                if v < a:
                    return v
                b = min(b, v)
            return v


        def alpha_beta(agent, depth, gameState, a, b):
            # terminal state, return state's utility
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)
            # agent = pacman, max
            if agent == 0:
                return ab_max_value(1, agent, depth, gameState, a, b)
            # agent != pacman; min
            else:
                # 0 pacman; 1; 2; 3;
                nextAgent = agent + 1
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0  #if nextAgent == 0:
                    depth += 1
                return ab_min_value(nextAgent, agent, depth, gameState, a, b)

        utility = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        for agentState in gameState.getLegalActions(0):
            v = alpha_beta(1, 0, gameState.generateSuccessor(0, agentState), alpha, beta)
            if v > utility:
                utility = v
                action = agentState
            if utility > beta:
                return utility
            alpha = max(alpha, utility)

        return action




###################################################
#         Expectimax Agent                       #
###################################################

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        # python autograder.py -q q4  --no-graphics
        # adversary which chooses amongst their getLegalActions uniformly at random.

        def exp_max_value(nextAgent, agent, depth, gameState):
            v = float("-inf")
            for newState in gameState.getLegalActions(agent):
                v = max(v, Expectimax(nextAgent, depth, gameState.generateSuccessor(agent, newState)))
            return v

        def exp_average(nextAgent, agent, depth, gameState):
            v = 0.
            pr = 1./ len(gameState.getLegalActions(agent))
            for newState in gameState.getLegalActions(agent):
                v = v + Expectimax(nextAgent, depth, gameState.generateSuccessor(agent, newState))
            return pr * v


        def Expectimax(agent, depth, gameState):
            # terminal state, return state's utility
            if gameState.isLose() or gameState.isWin() or depth == self.depth:
                return self.evaluationFunction(gameState)
            # agent = pacman, max
            if agent == 0:
                return exp_max_value(1, agent, depth, gameState)
            # agent != pacman; min
            else:
                # 0 pacman; 1; 2; 3;
                nextAgent = agent + 1
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0  #if nextAgent == 0:
                    depth += 1
                return exp_average(nextAgent, agent, depth, gameState)
                #min(value(nextAgent, depth, gameState.generateSuccessor(agent, newState)) for newState in gameState.getLegalActions(agent))


        maximum = float("-inf")
        for agentState in gameState.getLegalActions(0):
            utility = Expectimax(1, 0, gameState.generateSuccessor(0, agentState))
            if utility > maximum or maximum == float("-inf"):
                maximum = utility
                action = agentState

        return action

###################################################
#         better Evaluation Function           #
###################################################

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    # python autograder.py -q q5 --no-graphics
    # python autograder.py -q q5

    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    # if currentGameState.isLose():
    #     return float("-inf")
    # if currentGameState.isWin():
    #     return float("inf")

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    currentFoodList = newFood.asList()
    num_food = len(currentFoodList)
    #print(currentFoodList)
    # if currentFoodList == []:
    #     num_food = 0
    currentCapsuleList = currentGameState.getCapsules()  # newCapsule.asList()
    num_capsules = len(currentCapsuleList)

    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    GhostPositions = [Ghost.getPosition() for Ghost in newGhostStates]

    # if (newScaredTimes[0] > 0):
    #     return newScaredTimes[0] + currentGameState.getScore()

    def better_distance(p1, p2):
        M = manhattanDistance(p1, p2)
        # x1, x2 = sorted([p1[0], p2[0]])
        # y1, y2 = sorted([p1[1], p2[1]])
        # if x1 == x2:
        #     if currentGameState.hasWall(int(x1), int(y1 + 1)):
        #         return M + 2
        # if y1 == y2:
        #     if currentGameState.hasWall(int(x1 + 1), int(y1)):
        #         return M + 2
        return M

    time = 0
    kill_score = 0.0
    min_kill_dis = 9999
    for i in range(len(GhostPositions)):
        time = newScaredTimes[i]
        ghost_pos = GhostPositions[i]
        if time > 0:
            dis = better_distance(newPos, ghost_pos)
            if min_kill_dis > dis:
                min_kill_dis = dis
    if min_kill_dis < time and min_kill_dis !=0 :
        kill_score = 70/min_kill_dis




    # food:
    score_food = 0.0
    min_food_dis = 9999

    for food_pos in currentFoodList:
        dis = better_distance(food_pos, newPos)
        # if dis == 1:
        #     min_food_pos = food_pos
        #     min_food_dis = dis
        #     break
        if dis < min_food_dis:
            min_food_dis = dis
            min_food_pos = food_pos
        # print(min_food_dis)
    if num_food >= 1:
        score_food = 2.0/ min_food_dis
    # if num_food == 1:
    #     score_food = 2.0/ min_food_dis
    #     #print('1 left:', min_food_dis, score_food)
    if num_food == 0:
        score_food = 100
        #print('no food')


    score_cap = 0.0
    min_capsule_dis = 9999

    for i in currentCapsuleList:
        dis = better_distance(newPos, i)
        if dis < min_capsule_dis:
            min_capsule_dis = dis
    if num_capsules > 0:
        score_cap = 50.0/min_capsule_dis
    # print(min_food_dis)

    # print(newGhostStates[0])
    score_ghost = 0.0
    min_ghost_dis = 9999
    for ghost_state in currentGameState.getGhostPositions():
        g_dis = better_distance(newPos, ghost_state)
        if min_ghost_dis > g_dis:
            min_ghost_dis = g_dis
        if min_ghost_dis <= 1:
            score_ghost -= 9999
        else: # min_ghost_dis != 0:
            score_ghost -= 1.0/ min_ghost_dis

    return score_food + kill_score + score_cap + score_ghost - num_capsules*100 - num_food * 10 + currentGameState.getScore()

    #score_food + score_cap + score_ghost + currentGameState.getScore()


# python pacman.py -p ExpectimaxAgent -l mediumClassic -a evalFn=better,depth=3
# python pacman.py -p AlphaBetaAgent -l mediumClassic -a evalFn=better,depth=3
# python pacman.py -p AlphaBetaAgent -l mediumClassic -a evalFn=better,depth=3 -n 10 --frameTime 0
# python pacman.py -p ExpectimaxAgent -l mediumClassic -a evalFn=better,depth=3 -n 10 --frameTime 0

# Abbreviation
better = betterEvaluationFunction
