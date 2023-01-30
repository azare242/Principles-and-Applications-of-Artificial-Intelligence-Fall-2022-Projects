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

    def __init__(self, mdp, discount=0.9, iterations=100):
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
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        y = self.discount

        for _ in range(self.iterations):
            val = self.values.copy()
            for state in self.mdp.getStates():
                _max = float('-inf')
                for action in self.mdp.getPossibleActions(state):
                    t = float(0)
                    for state_prime, p in self.mdp.getTransitionStatesAndProbs(state, action):
                        r = self.mdp.getReward(state, action, state_prime)
                        v = val[state_prime]
                        t += p * (r + y * v)
                    if _max == float('-inf'):
                        _max = t
                    else:
                        _max = max(t, _max)
                if not self.mdp.isTerminal(state):
                    self.values[state] = _max
                else:
                    self.values[state] = float(0)

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q = float(0)
        y = self.discount
        for state_prime, p in self.mdp.getTransitionStatesAndProbs(state, action):
            r = self.mdp.getReward(state, action, state_prime)
            v = self.values[state_prime]
            q += p * (r + y * v)
        return q
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        q = []
        actions = []
        for action in self.mdp.getPossibleActions(state):
            q.append(self.getQValue(state, action))
            actions.append(action)
        if len(q) == 0:
            return None
        q_max = max(q)
        best_idxs = [idx for idx in range(len(q)) if q[idx] == q_max]
        return actions[best_idxs[-1]]

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
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

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        state_generator = loop_generator(self.mdp.getStates())
        for _ in range(self.iterations):
            state = next(state_generator)
            action = self.computeActionFromValues(state)
            if action is None:
                new_v = 0
            else:
                new_v = self.computeQValueFromValues(state, action)
            self.values[state] = new_v


def loop_generator(_list):
    while True:
        for x in _list:
            yield x


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):

        pq = util.PriorityQueue()
        temp = dict()

        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                for a in self.mdp.getPossibleActions(s):
                    for s_prime, _ in self.mdp.getTransitionStatesAndProbs(s, a):
                        if not s_prime in temp:
                            temp[s_prime] = set()
                        temp[s_prime].add(s)

        for s in self.mdp.getStates():
            if not self.mdp.isTerminal(s):
                q = [self.computeQValueFromValues(s, a) for a in self.mdp.getPossibleActions(s)]
                diff = abs(max(q) - self.values[s])
                pq.push(s, -1 * diff)

        for _ in range(self.iterations):
            if pq.isEmpty():
                break
            s = pq.pop()
            if not self.mdp.isTerminal(s):
                q = [self.computeQValueFromValues(s, a) for a in self.mdp.getPossibleActions(s)]
                self.values[s] = max(q)

                for p in temp[s]:
                    if not self.mdp.isTerminal(p):
                        q = [self.computeQValueFromValues(p, a) for a in self.mdp.getPossibleActions(p)]
                        diff = abs(max(q) - self.values[p])
                        if diff > self.theta:
                            pq.update(p, -1 * diff)
