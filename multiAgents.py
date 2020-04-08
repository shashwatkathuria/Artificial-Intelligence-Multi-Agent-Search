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

        # Getting information required
        successorScore = successorGameState.getScore()
        newFoodPositions = newFood.asList()

        # Variable for storing distance to closest food
        minDistFood = None

        # Looping over all food positions
        for foodPosition in newFoodPositions:
            # If the minimum distance food is not assigned yet
            if minDistFood == None:
                # Storing coordinates
                minDistFood = foodPosition
            # Otherwise if minimum distance is assigned
            else:
                # Getting present and new manhattan distances
                presentManhattan = manhattanDistance(minDistFood, newPos)
                newManhattan = manhattanDistance(foodPosition, newPos)
                # Comparing and storing if the new distance is closer than present
                if newManhattan < presentManhattan:
                    minDistFood = foodPosition

        # If food position is there
        if minDistFood == None:
            # Assigning a high constant for negative effect
            minDistFood = 0.2 * successorScore
        # Otherwise storing the actual distance value calculated from the coordinates
        # of minimum distance food
        else:
            minDistFood = manhattanDistance(minDistFood, newPos)

        # Variable for storing distance to closest ghost
        minDistGhost = None
        # Getting ghost positions
        newGhostPositions = successorGameState.getGhostPositions()

        # Looping over all ghost positions
        for ghostPosition in newGhostPositions:
            # If the minimum distance ghost is not assigned yet
            if minDistGhost == None:
                # Storing coordinates
                minDistGhost = ghostPosition
            # Otherwise if minimum distance is assigned
            else:
                # Getting present and new manhattan distances
                presentManhattan = manhattanDistance(minDistGhost, newPos)
                newManhattan = manhattanDistance(ghostPosition, newPos)
                # Comparing and storing if the new distance is closer than present
                if newManhattan < presentManhattan:
                    minDistGhost = ghostPosition

        # Storing the actual distance value calculated from the coordinates
        # of minimum distance ghost
        minDistGhost = manhattanDistance(minDistGhost, newPos)

        # Applying a very negative effect if ghost comes very close
        # to avoid the ghost
        if minDistGhost <= 1:
            minDistGhost = -10000

        # Returning final score
        return successorScore + minDistGhost - minDistFood

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

        # Helper recursive function for Minimax Algorithm
        def miniMax(totalAgents, agentIndex, currentGameState, depth):
            # Agent index to move in a cycle
            if totalAgents == agentIndex:
                agentIndex = 0
                # Increment depth after each agent moves once
                depth += 1

            # If the depth required is reached
            if depth == self.depth:
                # Returning value
                return self.evaluationFunction(currentGameState)

            # Getting possible actions for current agent
            possibleActions = currentGameState.getLegalActions(agentIndex)
            # If possible actions is an empty list
            if not possibleActions:
                # Returning value
                return self.evaluationFunction(currentGameState)

            # For Pacman
            if agentIndex == 0:
                
                # List to store values, maximum to be extracted at the end
                maxPossibilities = []

                # Looping over each action
                for action in possibleActions:
                    # Getting successor state of current action
                    successorGameState = currentGameState.generateSuccessor(agentIndex, action)
                    # Getting the value of mini (next agent value) through recursion, passing on to next agent
                    miniValue = miniMax(totalAgents, agentIndex + 1, successorGameState, depth)
                    # Appending the value and action
                    maxPossibilities.append([miniValue, action])

                # Returning the action leading to maximum value back to the caller
                if depth == 0:
                    return max(maxPossibilities)[1]
                # Returning the maximum value if depth still remaining to solve
                else:
                    return max(maxPossibilities)[0]

            # For Ghosts
            elif agentIndex > 0:

                # List to store values, minimum to be extracted at the end
                minPossibilities = []

                # Looping over each action
                for action in possibleActions:
                    # Getting successor state of current action
                    successorGameState = currentGameState.generateSuccessor(agentIndex, action)
                    # Getting the value of mini or max (next agent value) through recursion, passing on to next agent
                    nextValue = miniMax(totalAgents, agentIndex + 1, successorGameState, depth)
                    # Appending the value
                    minPossibilities.append(nextValue)

                # Returning the minimum value
                return min(minPossibilities)

        # Getting the total number of agents
        totalAgents = gameState.getNumAgents()
        # Calling function, storing final result
        finalResult = miniMax(totalAgents, 0, gameState, 0)
        # Returning final result
        return finalResult


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        # Helper recursive function for Alpha-Beta Pruning (Minimax Algorithm)
        def alphaBetaMiniMax(totalAgents, agentIndex, currentGameState, depth, alpha, beta):
            # Agent index to move in a cycle
            if totalAgents == agentIndex:
                agentIndex = 0
                # Increment depth after each agent moves once
                depth += 1

            # If the depth required is reached
            if depth == self.depth:
                # Returning value
                return self.evaluationFunction(currentGameState)

            # Getting possible actions for current agent
            possibleActions = currentGameState.getLegalActions(agentIndex)
            # If possible actions is an empty list
            if not possibleActions:
                # Returning value
                return self.evaluationFunction(currentGameState)

            # For Pacman
            if agentIndex == 0:

                # List to store values, maximum to be extracted at the end
                maxPossibilities = []
                # Variable to store the best value till running iteration
                bestValue = -9999999999

                # Looping over each action
                for action in possibleActions:
                    # Getting successor state of current action
                    successorGameState = currentGameState.generateSuccessor(agentIndex, action)
                    # Getting the value of mini (next agent value) through recursion, passing on to next agent
                    miniValue = alphaBetaMiniMax(totalAgents, agentIndex + 1, successorGameState, depth, alpha, beta)
                    # Appending the value and action
                    maxPossibilities.append([miniValue, action])
                    # Storing the best (maximum) value obtained till now
                    bestValue = max(bestValue, miniValue)

                    # Pruning (breaking early) if best value not useful
                    if bestValue > beta:
                        return bestValue

                    # Alpha to be the max of alpha till now and best value obtained till now
                    alpha = max(alpha, bestValue)

                # Returning the action leading to maximum value back to the caller
                if depth == 0:
                    return max(maxPossibilities)[1]
                # Returning the maximum value if depth still remaining to solve
                else:
                    return max(maxPossibilities)[0]

            # For Ghosts
            elif agentIndex > 0:

                # List to store values, minimum to be extracted at the end
                minPossibilities = []
                # Variable to store the best value till running iteration
                bestValue = 9999999999

                # Looping over each action
                for action in possibleActions:
                    # Getting successor state of current action
                    successorGameState = currentGameState.generateSuccessor(agentIndex, action)
                    # Getting the value of mini or max (next agent value) through recursion, passing on to next agent
                    nextValue = alphaBetaMiniMax(totalAgents, agentIndex + 1, successorGameState, depth, alpha, beta)
                    # Appending the value
                    minPossibilities.append(nextValue)
                    # Storing the best (minimum) value obtained till now
                    bestValue = min(bestValue, nextValue)

                    # Pruning (breaking early) if best value not useful
                    if bestValue < alpha:
                        return bestValue

                    # Beta to be the min of beta till now and best value obtained till now
                    beta = min(beta, bestValue)

                # Returning the minimum value
                return min(minPossibilities)

        # Getting the total number of agents
        totalAgents = gameState.getNumAgents()
        # Calling function, storing final result
        finalResult = alphaBetaMiniMax(totalAgents, 0, gameState, 0, -9999999999, 9999999999)

        # Returning final result
        return finalResult

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

        def expectiMax(totalAgents, agentIndex, currentGameState, depth):
            if totalAgents == agentIndex:
                agentIndex = 0
                depth += 1

            if depth == self.depth:
                return self.evaluationFunction(currentGameState)

            possibleActions = currentGameState.getLegalActions(agentIndex)
            if not possibleActions:
                return self.evaluationFunction(currentGameState)

            if agentIndex == 0:
                maxPossibilities = []
                for action in possibleActions:
                    successorGameState = currentGameState.generateSuccessor(agentIndex, action)
                    miniValue = expectiMax(totalAgents, agentIndex + 1, successorGameState, depth)
                    maxPossibilities.append([miniValue, action])

                if depth == 0:
                    return max(maxPossibilities)[1]
                else:
                    return max(maxPossibilities)[0]

            elif agentIndex > 0:

                randomValue = 0
                equalProbability = 1.0 / len(possibleActions)
                for action in possibleActions:
                    successorGameState = currentGameState.generateSuccessor(agentIndex, action)
                    nextValue = expectiMax(totalAgents, agentIndex + 1, successorGameState, depth)
                    randomValue += equalProbability * nextValue

                return randomValue

        totalAgents = gameState.getNumAgents()
        finalResult = expectiMax(totalAgents, 0, gameState, 0)

        return finalResult

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: Attracts pacman toward foods, pellets, more scared time and
                   keeps track of distance from closest ghost to avoid.
    """
    "*** YOUR CODE HERE ***"
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    currentGhostStates = currentGameState.getGhostStates()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
    currentFoodPositions = currentFood.asList()
    currentPellets = currentGameState.getCapsules()
    currentScore = currentGameState.getScore()

    extraScore = 0

    for pelletPosition in currentPellets:
        pelletManhattan = manhattanDistance(currentPos, pelletPosition)
        extraScore -= pelletManhattan

    for scaredTime in currentScaredTimes:
        extraScore += scaredTime

    for foodPosition in currentFoodPositions:
        foodManhattan = manhattanDistance(foodPosition, currentPos)
        extraScore -= foodManhattan

    minDistGhost = None
    currentGhostPositions = currentGameState.getGhostPositions()
    for ghostPosition in currentGhostPositions:
        if minDistGhost == None:
            minDistGhost = ghostPosition
        else:
            presentManhattan = manhattanDistance(minDistGhost, currentPos)
            newManhattan = manhattanDistance(ghostPosition, currentPos)
            if newManhattan < presentManhattan:
                minDistGhost = ghostPosition

    minDistGhost = manhattanDistance(minDistGhost, currentPos)
    if minDistGhost <= 1:
        extraScore -= 10000

    return currentScore + extraScore

# Abbreviation
better = betterEvaluationFunction
