# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

import time
def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    explored = set()  # non-identical set of closed node
    fringe = util.Stack()
    fringe.push((problem.getStartState(),[]))  #(node, [list: path]
    while not fringe.isEmpty():  # while True?
        #if (fringe == None): return failure
        popped = fringe.pop()
        #print('pooped:',popped)
        node = popped[0]
        solution = popped[1]
        if problem.isGoalState(node):
            break
        else:
            if node not in explored:
                explored.add(node)
                successors = problem.getSuccessors(node)
                for successor in successors:
                    #print(successor)
                    child_node = successor[0]
                    child_action = successor[1]
                    path_actions = solution + [child_action]
                    fringe.push((child_node, path_actions))
                    #print('full path:', full_path)
                    #time.sleep(10)
    return solution
    # mediumMaze should have a length of 130
    # util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    fringe = util.Queue()
    explored = set()
    fringe.push((problem.getStartState(),[]))
    while not fringe.isEmpty(): # while True?
        popped = fringe.pop()
        node = popped[0]
        solution = popped[1]
        if problem.isGoalState(node):
            break
        else:
            if node not in explored:
                explored.add(node)
                successors = problem.getSuccessors(node)
                for successor in successors:
                    child_node = successor[0]
                    child_action = successor[1]
                    path_actions = solution + [child_action]
                    fringe.push((child_node, path_actions))
                    # print('full path:', full_path)
                    # time.sleep(10)
    return solution
    #util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    # Dijkstra's algorithm
    fringe = util.PriorityQueue()
    explored = set()
    fringe.push((problem.getStartState(), [], 0), 0)
    # ( (state, action: list, cost), priority )
    while not fringe.isEmpty():
        node, solution, existing_cost = fringe.pop()
        # solution
        # existing_cost
        if problem.isGoalState(node):
            break
        else:
            if node not in explored:
                explored.add(node)
                successors = problem.getSuccessors(node)
                for successor in successors:
                    child_node = successor[0]
                    child_action = successor[1]
                    child_cost = successor[2]
                    path_actions = solution + [child_action]
                    full_cost = existing_cost + child_cost
                    fringe.push((child_node, path_actions, full_cost), full_cost)
    return solution
    # very low and very high path costs for the StayEastSearchAgent and StayWestSearchAgent respectively
    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic): #
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # heuristic = manhattanHeuristic # in searchAgent.py
    # manhattanHeuristic(position, problem, info={}):
    fringe = util.PriorityQueue()
    explored = set()
    heuristic_w_cost = heuristic(problem.getStartState(), problem) + 0
    fringe.push((problem.getStartState(), [], 0), heuristic_w_cost)
    # ( (state, action: list, cost), priority )
    while not fringe.isEmpty():
        node, solution, existing_cost = fringe.pop()
        # solution
        # existing_cost
        if problem.isGoalState(node):
            break
        else:
            if node not in explored:
                explored.add(node)
                successors = problem.getSuccessors(node)
                for successor in successors:
                    child_node = successor[0]
                    child_action = successor[1]
                    child_cost = successor[2]
                    path_actions = solution + [child_action]
                    full_cost = existing_cost + child_cost
                    child_heuristic = heuristic(child_node, problem)
                    fringe.push((child_node, path_actions, full_cost), full_cost + child_heuristic)
    return solution
    # slightly faster than uniform cost search (about 549 vs. 620 search nodes expanded in our implementation
    #util.raiseNotDefined()



################
class StateInfo:
    def __init__(self, parent_info, state, direction, other=None):
        self.parent_info = parent_info
        self.state = state
        self.direction = direction
        self.other = other

    # def __init__(self, other_info):
    #     self.parent_info = other_info.parent_info
    #     self.state = other_info.state
    #     self.direction = other_info.direction

    def get_path(self):
        res = []
        info = self
        while info.parent_info is not None:
            res.append(info.direction)
            info = info.parent_info
        res.reverse()
        return res

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
