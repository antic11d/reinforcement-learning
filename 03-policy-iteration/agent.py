import util, random
import numpy as np
from copy import deepcopy


class Agent:

    def getAction(self, state):
        """
        For the given state, get the agent's chosen
        action.  The agent knows the legal actions
        """
        abstract

    def getValue(self, state):
        """
        Get the value of the state.
        """
        abstract

    def getQValue(self, state, action):
        """
        Get the q-value of the state action pair.
        """
        abstract

    def getPolicy(self, state):
        """
        Get the policy recommendation for the state.

        May or may not be the same as "getAction".
        """
        abstract

    def update(self, state, action, nextState, reward):
        """
        Update the internal state of a learning agent
        according to the (state, action, nextState)
        transistion and the given reward.
        """
        abstract


class RandomAgent(Agent):
    """
    Clueless random agent, used only for testing.
    """

    def __init__(self, actionFunction):
        self.actionFunction = actionFunction

    def getAction(self, state):
        return random.choice(self.actionFunction(state))

    def getValue(self, state):
        return 0.0

    def getQValue(self, state, action):
        return 0.0

    def getPolicy(self, state):
        return 'random'

    def update(self, state, action, nextState, reward):
        pass


################################################################################
class ValueIterationAgent(Agent):

  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
    Your value iteration agent should take an mdp on
    construction, run the indicated number of iterations
    and then act according to the resulting policy.
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations

    states = self.mdp.getStates()
    number_states = len(states)

    self.V = { s: 0 for s in states }


    for i in range(iterations):
      newV = {}
      for s in states:
        actions = self.mdp.getPossibleActions(s)
        if len(actions) < 1:
          newV[s] = 0.0
        else:
          newV[s] = np.max([ self.mdp.getReward(s, a, None) + \
                             self.discount * np.sum([prob * self.V[sp]
                                                     for sp, prob in self.mdp.getTransitionStatesAndProbs(s, a) ])
                             for a in actions ])
      self.V = newV


  def getValue(self, state):
    """
    Look up the value of the state (after the indicated
    number of value iteration passes).
    """
    return self.V[state]



  def getQValue(self, state, action):
    """
    Look up the q-value of the state action pair
    (after the indicated number of value iteration
    passes).  Note that value iteration does not
    necessarily create this quantity and you may have
    to derive it on the fly.
    """
    # get all successor states and probabilities and evaluate value of these states
    return self.mdp.getReward(state, action, None) + \
      self.discount *  np.sum([self.V[sp] * prob for sp, prob in self.mdp.getTransitionStatesAndProbs(state, action)])

  def getPolicy(self, state):
    """
    Look up the policy's recommendation for the state
    (after the indicated number of value iteration passes).
    """
    # do greedy on Q
    actions = self.mdp.getPossibleActions(state)
    if len(actions) < 1:
      return None
    else:
      qValues = [self.getQValue(state, a) for a in actions]
      action_index = np.argmax(qValues)
      return actions[action_index]

  def getAction(self, state):
    """
    Return the action recommended by the policy.
    """
    return self.getPolicy(state)


  def update(self, state, action, nextState, reward):
    """
    Not used for value iteration agents!
    """
    pass

class PolicyIterationAgent(Agent):

  def __init__(self, mdp, discount = 0.9, iterations = 100):
    """
    Your policy iteration agent should take an mdp on
    construction, run the indicated number of iterations
    and then act according to the resulting policy.
    """
    self.mdp = mdp
    self.discount = discount
    self.iterations = iterations

    states = self.mdp.getStates()
    number_states = len(states)

    self.V = { s:0 for s in states }
    self.pi = { s:self.mdp.getPossibleActions(s)[-1] if self.mdp.getPossibleActions(s) else None for s in states }

    counter = 0

    while True:
        # Policy evaluation
        for i in range(iterations):
            newV={}
            for s in states:
                a = self.pi[s]
                if a is None:
                    newV[s] = 0.0
                else:
                    newV[s] = self.mdp.getReward(s,a, None) + \
                                       self.discount * np.sum([ prob*self.V[sp]
                                                                for sp,prob in self.mdp.getTransitionStatesAndProbs(s,a) ])

            self.V = newV

        # Policy improvement

        policy_stable = True
        for s in states:
            actions = self.mdp.getPossibleActions(s)
            if len(actions) < 1:
                self.pi[s] = None
            else:
                old_action = self.pi[s]
                self.pi[s] = actions[np.argmax([ self.mdp.getReward(s,a, None) + \
                                      self.discount * np.sum([ prob*self.V[sp]
                                                               for sp,prob in self.mdp.getTransitionStatesAndProbs(s,a) ])
                                      for a in actions])]
                policy_stable = policy_stable and old_action == self.pi[s]

        counter += 1

        if policy_stable: break

    print("Policy converged after %i iterations of policy iteration" % counter)

  def getValue(self, state):
    """
    Look up the value of the state (after the policy converged).
    """
    return self.V[state]



  def getQValue(self, state, action):
    """
    Look up the q-value of the state action pair
    (after the indicated number of value iteration
    passes).  Note that policy iteration does not
    necessarily create this quantity and you may have
    to derive it on the fly.
    """
    # get all successor states and probabilties and evaluate value of these states
    return self.mdp.getReward(state, action, None) + \
      self.discount *  np.sum([self.V[sp] * prob for sp, prob in self.mdp.getTransitionStatesAndProbs(state, action)])

  def getPolicy(self, state):
    """
    Look up the policy's recommendation for the state
    (after the indicated number of value iteration passes).
    """
    return self.pi[state]

  def getAction(self, state):
    """
    Return the action recommended by the policy.
    """
    return self.getPolicy(state)


  def update(self, state, action, nextState, reward):
    """
    Not used for policy iteration agents!
    """
    pass


################################################################################
# Below can be ignored for Exercise 7

class QLearningAgent(Agent):

    def __init__(self, actionFunction, discount=0.9, learningRate=0.1, epsilon=0.2):
        """
        A Q-Learning agent gets nothing about the mdp on
        construction other than a function mapping states to actions.
        The other parameters govern its exploration
        strategy and learning rate.
        """
        self.setLearningRate(learningRate)
        self.setEpsilon(epsilon)
        self.setDiscount(discount)
        self.actionFunction = actionFunction

        raise "Your code here."

    # THESE NEXT METHODS ARE NEEDED TO WIRE YOUR AGENT UP TO THE CRAWLER GUI

    def setLearningRate(self, learningRate):
        self.learningRate = learningRate

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def setDiscount(self, discount):
        self.discount = discount

    # GENERAL RL AGENT METHODS

    def getValue(self, state):
        """
        Look up the current value of the state.
        """

        raise ValueError("Your code here.")

    def getQValue(self, state, action):
        """
        Look up the current q-value of the state action pair.
        """

        raise ValueError("Your code here.")

    def getPolicy(self, state):
        """
        Look up the current recommendation for the state.
        """

        raise ValueError("Your code here.")

    def getAction(self, state):
        """
        Choose an action: this will require that your agent balance
        exploration and exploitation as appropriate.
        """

        raise ValueError("Your code here.")

    def update(self, state, action, nextState, reward):
        """
        Update parameters in response to the observed transition.
        """

        raise ValueError("Your code here.")
