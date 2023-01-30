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
import math

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

import sys


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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

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
        new_score = successorGameState.getScore() * 5
        new_food_loc = newFood.asList()
        foods_current = len(currentGameState.getFood().asList())
        foods_new = len(new_food_loc)

        power_dots_loc = successorGameState.getCapsules()
        num_power_dots_current = len(currentGameState.getCapsules())
        num_power_dots_new = len(power_dots_loc)

        if num_power_dots_current > num_power_dots_new or len(power_dots_loc) == 0:
            power_dot_dist = -100
        else:
            if len(power_dots_loc) != 0:
                power_dot_dist = min_manhattan_dist(newPos, power_dots_loc)
            else:
                power_dot_dist = 0

        if num_power_dots_current > foods_current or len(new_food_loc) == 0:
            food_dist = -50
        else:
            if len(new_food_loc) != 0:
                food_dist = min_manhattan_dist(newPos, new_food_loc) / 2
            else:
                food_dist = 0
        if len(newGhostStates) != 0:
            ghost_to_run_away_from_dist = None
            ghost_to_eat_dist = None
            if newScaredTimes == [0 for ghost in newGhostStates]:
                ghost_to_run_away_from_dist = sys.maxsize
                for ghost in newGhostStates:
                    t = manhattanDistance(ghost.getPosition(), newPos)
                    if t < ghost_to_run_away_from_dist:
                        ghost_to_run_away_from_dist = t
            if ghost_to_run_away_from_dist == 1:
                return -999999999  # i use sys.maxsize but i got error
            else:
                ghosts_num_current = len(currentGameState.getGhostStates())
                ghosts_num_new = len(newGhostStates)
                for ghost in newGhostStates:
                    ghost_dist = manhattanDistance(ghost.getPosition(), newPos)
                    if ghost.scaredTimer <= 1:
                        if ghost_dist == 1:
                            return sys.maxsize * -1
                        if ghost_to_run_away_from_dist is None or ghost_to_run_away_from_dist > ghost_dist:
                            ghost_to_run_away_from_dist = ghost_dist
                    else:
                        if ghosts_num_current > ghosts_num_new:
                            ghost_to_eat_dist = -50
                        elif ghost_to_eat_dist is None or ghost_to_eat_dist > ghost_dist:
                            ghost_to_eat_dist = ghost_dist
            if ghost_to_eat_dist is not None:
                new_score -= ghost_to_eat_dist
            if ghost_to_run_away_from_dist is not None:
                new_score += math.log1p(1 + ghost_to_run_away_from_dist)
        if action == 'Stop':
            if new_score > 0:
                new_score /= 2
            else:
                new_score *= 2

        return new_score - (food_dist + power_dot_dist)


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


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
        pacman_idx = 0
        moves = gameState.getLegalActions(pacman_idx)

        states = [gameState.generateSuccessor(pacman_idx, action) for action in moves]
        scores = [get_scores_minimax(self, state, 1, 0) for state in states]
        best_score = max(scores)
        best_idxs = [idx for idx in range(len(scores)) if scores[idx] == best_score]
        return moves[best_idxs[-1]]

        util.raiseNotDefined()


def get_scores_minimax(minimaxAgent, gameState, idx, depth):
    if idx == gameState.getNumAgents():
        depth += 1
    idx %= gameState.getNumAgents()
    if depth == minimaxAgent.depth or len(
            gameState.getLegalActions(idx)) == 0 or gameState.isWin() or gameState.isLose():
        return minimaxAgent.evaluationFunction(gameState)

    next_state = [gameState.generateSuccessor(idx, action) for action in gameState.getLegalActions(idx)]
    scores = [get_scores_minimax(minimaxAgent, state, idx + 1, depth) for state in next_state]

    if idx > 0:
        return min(scores)
    else:
        return max(scores)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        b = sys.maxsize
        a = -1 * b
        pacman_idx = 0
        moves = gameState.getLegalActions(pacman_idx)
        scores = []

        for action in moves:
            t = get_scores_a_b(self, gameState.generateSuccessor(pacman_idx, action), 1, 0, a, b)
            scores.append(t)
            a = max(a, t)

        best_score = max(scores)
        best_idxs = [idx for idx in range(len(scores)) if scores[idx] == best_score]
        return moves[best_idxs[-1]]
        util.raiseNotDefined()


def get_scores_a_b(alphaBetaAgent, gameState, idx, depth, a, b):
    if idx == gameState.getNumAgents():
        depth += 1
    idx %= gameState.getNumAgents()
    moves = gameState.getLegalActions(idx)
    if depth == alphaBetaAgent.depth or len(moves) == 0 or gameState.isWin() or gameState.isLose():
        return alphaBetaAgent.evaluationFunction(gameState)

    if idx > 0:
        res = sys.maxsize
        for action in moves:
            t_score = get_scores_a_b(alphaBetaAgent, gameState.generateSuccessor(idx, action), idx + 1, depth, a, b)
            res = min(res, t_score)
            if res < a:
                return res;
            b = min(b, res)
        return res
    else:
        res = -1 * sys.maxsize
        for action in moves:
            t_score = get_scores_a_b(alphaBetaAgent, gameState.generateSuccessor(idx, action), idx + 1, depth, a, b)
            res = max(res, t_score)
            if res > b:
                return res
            a = max(a, res)
        return res


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
        pacman_idx = 0
        moves = gameState.getLegalActions(pacman_idx)
        states = [gameState.generateSuccessor(pacman_idx, action) for action in moves]

        scores = [get_scores_expectimax(self, state, 1, 0) for state in states]
        best_scores = max(scores)
        best_idxs = [idx for idx in range(len(scores)) if scores[idx] == best_scores]
        return moves[best_idxs[-1]]
        util.raiseNotDefined()


def avg(_list):
    n = float(len(_list))
    sum = 0
    for i in _list:
        sum += i
    return sum / n


def get_scores_expectimax(expectimaxAgent, gameState, idx, depth):
    if idx == gameState.getNumAgents():
        depth += 1

    idx %= gameState.getNumAgents()

    if depth == expectimaxAgent.depth or gameState.getLegalActions(
            idx) == 0 or gameState.isWin() or gameState.isLose():
        return expectimaxAgent.evaluationFunction(gameState)

    next_states = [gameState.generateSuccessor(idx, action) for action in gameState.getLegalActions(idx)]
    next_scores = [get_scores_expectimax(expectimaxAgent, state, idx + 1, depth) for state in next_states]

    if idx > 0:
        return avg(next_scores)
    else:
        return max(next_scores)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    Don't forget to use pacmanPosition, foods, scaredTimers, ghostPositions!
    DESCRIPTION: <write something here so we know what you did>
    """

    pacmanPosition = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPositions = currentGameState.getGhostPositions()

    "*** YOUR CODE HERE ***"
    score_current = currentGameState.getScore() * 5
    if currentGameState.isWin():
        return 999999999999999  # Alireza Zare : i try sys.maxsize but i got errors
    if currentGameState.isLose():
        return -999999999999999 + abs(score_current)

    power_dot_loc = currentGameState.getCapsules()
    if len(power_dot_loc) != 0:
        power_dot_dist = min_manhattan_dist(pacmanPosition, power_dot_loc)
    else:
        power_dot_dist = 0
    foods_list = foods.asList()
    food_dist = min_manhattan_dist(pacmanPosition, foods_list) / 2

    ghost_to_run_away_from_dist = None
    ghost_to_eat_dist = None

    if len(ghostStates) != 0:
        if scaredTimers == [0 for _ in scaredTimers]:
            ghost_to_run_away_from_dist = min_manhattan_dist(pacmanPosition, ghostPositions)
        else:
            for ghost in ghostStates:
                ghost_dist = manhattanDistance(pacmanPosition, ghost.getPosition())
                if ghost.scaredTimer <= 1:
                    if ghost_to_run_away_from_dist is None or ghost_to_run_away_from_dist > ghost_dist:
                        ghost_to_run_away_from_dist = ghost_dist
                else:
                    if ghost_to_eat_dist is None or ghost_to_eat_dist > ghost_dist:
                        ghost_to_eat_dist = ghost_dist

    if ghost_to_eat_dist is not None:
        score_current -= ghost_to_eat_dist
    if ghost_to_run_away_from_dist is not None:
        score_current += math.log1p(1 + ghost_to_run_away_from_dist)

    return score_current - (food_dist + power_dot_dist)
    util.raiseNotDefined()


def min_manhattan_dist(pos, loc):
    minimum = sys.maxsize
    for _loc in loc:
        t = manhattanDistance(_loc, pos)
        if t < minimum:
            minimum = t
    return minimum


# Abbreviation
better = betterEvaluationFunction
