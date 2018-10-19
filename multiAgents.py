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


from util import manhattanDistance
from game import Directions
import random, util
import sys

from game import Agent

class ReflexAgent(Agent):
    _expanded = 0
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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        prevFood = gameState.getFood()
        oldFoodList = prevFood.asList()
        if len(oldFoodList) == 1:
            stats = gameState.generatePacmanSuccessor(legalMoves[chosenIndex])
            newPos = stats.getPacmanPosition()
            if newPos in oldFoodList:
                print("Nodes expanded: {}".format(self._expanded + len(legalMoves)))

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

        defScore = successorGameState.getScore()
        prevFood = currentGameState.getFood()
        oldFoodList = prevFood.asList()
        currentPos = successorGameState.getPacmanPosition()
        score = 99999
        self._expanded += 1

        #Give stio lowest priority
        if action == 'Stop':
            return -100

        for state in newGhostStates: #When you come across a ghost, return lowest priority
            if state.getPosition() == currentPos: #If power pellet is not eaten
                if state.scaredTimer == 0:
                    return -100
                else:
                    return 100 #Attack the ghost if power pellet is eaten

        for item in oldFoodList: #Direct Pacman towards food
            distance = (manhattanDistance(item, currentPos))

            if (distance < score):
                score = distance

        return (-1 * score)

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
        """
        "*** YOUR CODE HERE ***"

        # minimax does DFS search
        # recursive solution?
        # multiple min layers for every ghost
        # one max layer for pacman agent

        numOfAgents = gameState.getNumAgents()
        scores = []

        def wrapper(gameState, agentIndex):
          # reach leaf node or end or reaches base case
          if agentIndex>=self.depth*numOfAgents or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

          if agentIndex%numOfAgents:
            # ghost
            # one min layer for every ghost agent
            minply = sys.maxint
            for nextAction in list(filter(lambda x: x!='Stop', gameState.getLegalActions(agentIndex%numOfAgents))):
              nextGameState = gameState.generateSuccessor(agentIndex%numOfAgents, nextAction)
              # recursively find the minimum
              minply = min(minply, wrapper(nextGameState, agentIndex + 1))
            return minply
          else:
            # pacman agent
            # max layer
            maxply = -sys.maxint
            for nextAction in list(filter(lambda x: x!='Stop', gameState.getLegalActions(agentIndex%numOfAgents))):
              nextGameState = gameState.generateSuccessor(agentIndex%numOfAgents, nextAction)
              maxply = max(maxply, wrapper(nextGameState, agentIndex + 1))
              if agentIndex == 0:
                # score of each legal pacman action
                scores.append(maxply)
            return maxply

        wrapper(gameState, 0)

        return list(filter(lambda x: x!='Stop', gameState.getLegalActions(0))) [scores.index(max(scores))]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numOfAgents = gameState.getNumAgents()
        scores = []

        def wrapper(gameState, agentIndex, alpha, beta):
          # reach leaf node or end or reaches base case
          if agentIndex>=self.depth*numOfAgents or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

          if agentIndex%numOfAgents:
            # ghost
            # one min layer for every ghost agent
            minply = sys.maxint
            for nextAction in list(filter(lambda x: x!='Stop', gameState.getLegalActions(agentIndex%numOfAgents))):
              nextGameState = gameState.generateSuccessor(agentIndex%numOfAgents, nextAction)
              minply = min(minply, wrapper(nextGameState, agentIndex + 1, alpha, beta))
              beta = min(beta, minply)
              #Beta Pruning
              if beta < alpha:
                break
            return minply
          else:
            # pacman agent
            # max layer
            maxply = -sys.maxint
            for nextAction in list(filter(lambda x: x!='Stop', gameState.getLegalActions(agentIndex%numOfAgents))):
              nextGameState = gameState.generateSuccessor(agentIndex%numOfAgents, nextAction)
              maxply = max(maxply, wrapper(nextGameState, agentIndex + 1, alpha, beta))
              alpha = max(alpha, maxply)
              if agentIndex == 0:
                # score of each legal pacman action
                scores.append(maxply)
                #Alpha pruning
              if beta < alpha:
                break
            return maxply

        wrapper(gameState, 0, -sys.maxint, sys.maxint)

        return list(filter(lambda x: x!='Stop', gameState.getLegalActions(0))) [scores.index(max(scores))]

        util.raiseNotDefined()

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

        numOfAgents = gameState.getNumAgents()
        scores = []

        def wrapper(gameState, agentIndex):
          # reach leaf node or end or reaches base case
          if agentIndex>=self.depth*numOfAgents or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

          if agentIndex%numOfAgents:
            # ghost
            # one min layer for every ghost agent
            minply = sys.maxint
            successorScores = []
            for nextAction in list(filter(lambda x: x!='Stop', gameState.getLegalActions(agentIndex%numOfAgents))):
              nextGameState = gameState.generateSuccessor(agentIndex%numOfAgents, nextAction)
              # recursively find the minimum
              minply = wrapper(nextGameState, agentIndex + 1)
              successorScores.append(minply)
            #Averaging the scores of all the legal actions of the Ghost
            avgScore = float(sum(successorScores))/float(len(successorScores))
            return avgScore
          else:
            # pacman agent
            # max layer
            maxply = -sys.maxint
            for nextAction in list(filter(lambda x: x!='Stop', gameState.getLegalActions(agentIndex%numOfAgents))):
              nextGameState = gameState.generateSuccessor(agentIndex%numOfAgents, nextAction)
              maxply = max(maxply, wrapper(nextGameState, agentIndex + 1))
              if agentIndex == 0:
                # score of each legal pacman action
                scores.append(maxply)
            return maxply

        wrapper(gameState, 0)

        return list(filter(lambda x: x!='Stop', gameState.getLegalActions(0))) [scores.index(max(scores))]

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
