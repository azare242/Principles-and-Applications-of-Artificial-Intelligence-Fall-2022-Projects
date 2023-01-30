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
    return [s, s, w, s, w, w, s, w]


def updateFringe(fringe, unvisited_nodes, current_list):
    for neighbour_state in unvisited_nodes:
        tl = [] + current_list
        tl.append(neighbour_state)
        fringe.push(tl)

def answer(path):
    ans = []
    for _i in range(len(path) - 1):
        ts = path[_i + 1]
        ans.append(ts[1])
    return ans
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
    current_state = problem.getStartState()
    fringe = util.Stack()
    fringe.push([[current_state]])
    visited_nodes = []

    while not fringe.isEmpty():

        current_list = fringe.pop()
        current_state = current_list[len(current_list) - 1]

        if current_state[0] not in visited_nodes:
            visited_nodes.append(current_state[0])
            current_successors = problem.getSuccessors(current_state[0])

            unvisited_nodes = []

            for _i in range(len(current_successors)):
                if current_successors[_i][0] not in visited_nodes:
                    unvisited_nodes.append(current_successors[_i])

            if len(unvisited_nodes) > 0:
                last_neighbour = unvisited_nodes[len(unvisited_nodes) - 1]

                if problem.isGoalState(last_neighbour[0]):
                    return answer(current_list + [last_neighbour])

            updateFringe(fringe, unvisited_nodes, current_list)

    util.raiseNotDefined()

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    current_state = problem.getStartState()
    fringe = util.Queue()
    fringe.push([(current_state, None, 0)])
    visited_nodes = []

    while not fringe.isEmpty():
        current_list = fringe.pop()
        current_state = current_list[len(current_list) - 1]
        if current_state[0] not in visited_nodes:
            visited_nodes.append(current_state[0])

            if problem.isGoalState(current_state[0]):
                return answer(current_list)
            current_successor = problem.getSuccessors(current_state[0])
            unvisited_nodes = []

            for _i in range(len(current_successor)):
                if current_successor[_i][0] not in visited_nodes:
                    unvisited_nodes.append(current_successor[_i])

            updateFringe(fringe, unvisited_nodes, current_list)

    util.raiseNotDefined()


def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    current_state = problem.getStartState()
    fringe = util.PriorityQueue()
    fringe.push([[current_state, None, 0]], 0)
    visited_nodes = []

    while not fringe.isEmpty():
        current_list = fringe.pop()
        current_state = current_list[len(current_list) - 1]

        if current_state[0] not in visited_nodes:
            visited_nodes.append(current_state[0])

            if problem.isGoalState(current_state[0]):
                return answer(current_list)

            current_successor = problem.getSuccessors(current_state[0])
            unvisited_nodes = []
            for _i in range(len(current_successor)):
                if current_successor[_i][0] not in visited_nodes:
                    unvisited_nodes.append(current_successor[_i])

            current_uniform_cost = 0
            for _i in range(len(current_list)):
                current_uniform_cost += current_list[_i][2]

            for neighbour_state in unvisited_nodes:
                tl = [] + current_list
                tl.append(neighbour_state)
                tuc = current_uniform_cost + neighbour_state[2]
                fringe.push(tl,tuc)

    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    current_state = problem.getStartState()
    fringe = util.PriorityQueue()
    fringe.push([(current_state, None, 0)], heuristic(current_state, problem))
    visited_nodes = []

    while not fringe.isEmpty():
        current_list = fringe.pop()
        current_state = current_list[len(current_list) - 1]


        if current_state[0] not in visited_nodes:
            visited_nodes.append(current_state[0])

            if problem.isGoalState(current_state[0]):
                return answer(current_list)

            current_successors = problem.getSuccessors(current_state[0])
            unvisited_nodes = []
            for _i in range(len(current_successors)):
                if current_successors[_i][0] not in visited_nodes:
                    unvisited_nodes.append(current_successors[_i])

            current_uniformed_cost = 0
            for _i in range(len(current_list)):
                current_uniformed_cost = current_uniformed_cost + current_list[_i][2]

            for neighbour_state in unvisited_nodes:
                tl = [] + current_list
                tl.append(neighbour_state)

                tc = current_uniformed_cost + neighbour_state[2] + heuristic(neighbour_state[0], problem)
                fringe.push(tl, tc)

    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
