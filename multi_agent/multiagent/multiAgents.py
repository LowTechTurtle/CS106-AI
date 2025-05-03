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
from game import Directions, Actions
import random, util
import time

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
        #return successorGameState.getScore()
        food = currentGameState.getFood()
        currentPos = list(successorGameState.getPacmanPosition())
        distance = float("-Inf")

        foodList = food.asList()

        if action == 'Stop':
            return float("-Inf")

        for state in newGhostStates:
            if state.getPosition() == tuple(currentPos) and (state.scaredTimer == 0):
                return float("-Inf")

        for x in foodList:
            tempDistance = -1 * (manhattanDistance(currentPos, x))
            if (tempDistance > distance):
                distance = tempDistance

        return distance

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
#        begin = time.time()
        def minimax(state):
            bestValue, bestAction = None, None
            print(state.getLegalActions(0))
            value = []
            for action in state.getLegalActions(0):
                #value = max(value,minValue(state.generateSuccessor(0, action), 1, 1))
                succ  = minValue(state.generateSuccessor(0, action), 1, 1)
                value.append(succ)
                if bestValue is None:
                    bestValue = succ
                    bestAction = action
                else:
                    if succ > bestValue:
                        bestValue = succ
                        bestAction = action
            print(value)
            return bestAction

        def minValue(state, agentIdx, depth):
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = min(value, succ)

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)


        def maxValue(state, agentIdx, depth):
            if depth > self.depth:
                return self.evaluationFunction(state)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth)
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)
                
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        action = minimax(gameState)
#        done = time.time()
#        print(f"Get Action time: {done-begin}")
        return action

        # def minimax_search(state, agentIndex, depth):
        #     # if in min layer and last ghost
        #     if agentIndex == state.getNumAgents():
        #         # if reached max depth, evaluate state
        #         if depth == self.depth:
        #             return self.evaluationFunction(state)
        #         # otherwise start new max layer with bigger depth
        #         else:
        #             return minimax_search(state, 0, depth + 1)
        #     # if not min layer and last ghost
        #     else:
        #         moves = state.getLegalActions(agentIndex)
        #         # if nothing can be done, evaluate the state
        #         if len(moves) == 0:
        #             return self.evaluationFunction(state)
        #         # get all the minimax values for the next layer with each node being a possible state after a move
        #         next = (minimax_search(state.generateSuccessor(agentIndex, m), agentIndex + 1, depth) for m in moves)

        #         # if max layer, return max of layer below
        #         if agentIndex == 0:
        #             return max(next)
        #         # if min layer, return min of layer below
        #         else:
        #             return min(next)
        # # select the action with the greatest minimax value
        # result = max(gameState.getLegalActions(0), key=lambda x: minimax_search(gameState.generateSuccessor(0, x), 1, 1))

        # return result        

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
#        begin = time.time()
        def minimax(state):
            bestValue, bestAction = None, None
            print(state.getLegalActions(0))
            value = []
            alpha = float("-Inf")
            beta = float("Inf")
            for action in state.getLegalActions(0):
                #value = max(value,minValue(state.generateSuccessor(0, action), 1, 1))
                succ  = minValue(state.generateSuccessor(0, action), 1, 1, alpha, beta)
                value.append(succ)
                if bestValue is None:
                    bestValue = succ
                    bestAction = action
                else:
                    if succ > bestValue:
                        bestValue = succ
                        bestAction = action
            print(value)
            return bestAction

        def minValue(state, agentIdx, depth, alpha, beta):
            if agentIdx == state.getNumAgents():
                return maxValue(state, 0, depth + 1, alpha, beta)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth, alpha, beta)
                if value is None:
                    value = succ
                else:
                    value = min(value, succ)
                if value <= alpha:
                    return value
                beta = min(beta, value)
            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)


        def maxValue(state, agentIdx, depth, alpha, beta):
            if depth > self.depth:
                return self.evaluationFunction(state)
            value = None
            for action in state.getLegalActions(agentIdx):
                succ = minValue(state.generateSuccessor(agentIdx, action), agentIdx + 1, depth, alpha, beta)
                if value is None:
                    value = succ
                else:
                    value = max(value, succ)
                if value >= beta:
                    return value
                alpha = max(alpha, value)

            if value is not None:
                return value
            else:
                return self.evaluationFunction(state)

        action = minimax(gameState)
#        done = time.time()
#        print(f"Get Action time: {done-begin}")
        return action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """

        def expectimax(state, agentIdx, depth):
            if depth == self.depth or state.isWin() or state.isLose():
                return self.evaluationFunction(state)

            if agentIdx == 0:  # Pacman (max)
                return maxValue(state, agentIdx, depth)
            else:  # Ghost (expectation)
                return expValue(state, agentIdx, depth)

        def maxValue(state, agentIdx, depth):
            bestValue = float('-inf')
            bestAction = None
            for action in state.getLegalActions(agentIdx):
                successor = state.generateSuccessor(agentIdx, action)
                value = expectimax(successor, agentIdx + 1, depth)
                if value > bestValue:
                    bestValue = value
                    bestAction = action
            if depth == 0:  # If root node, return action
                return bestAction
            return bestValue

        def expValue(state, agentIdx, depth):
            actions = state.getLegalActions(agentIdx)
            if not actions:
                return self.evaluationFunction(state)
            total = 0
            prob = 1.0 / len(actions)
            for action in actions:
                successor = state.generateSuccessor(agentIdx, action)
                if agentIdx == state.getNumAgents() - 1:
                    value = expectimax(successor, 0, depth + 1)
                else:
                    value = expectimax(successor, agentIdx + 1, depth)
                total += prob * value
            return total

        return expectimax(gameState, 0, 0)


def closestItemDistance(currentGameState, items):
    """Returns the maze distance to the closest item present in items"""

    # BFS to find the maze distance from position to closest item
    walls = currentGameState.getWalls()

    start = currentGameState.getPacmanPosition()

    # Dictionary storing the maze distance from start to any given position
    distance = {start: 0}

    # Set of visited positions in order to avoid revisiting them again
    visited = {start}

    queue = util.Queue()
    queue.push(start)

    while not queue.isEmpty():

        position = x, y = queue.pop()

        if position in items: return distance[position]

        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:

            dx, dy = Actions.directionToVector(action)
            next_position = nextx, nexty = int(x + dx), int(y + dy)

            if not walls[nextx][nexty] and next_position not in visited:
                queue.push(next_position)
                visited.add(next_position)
                # A single action separates position from next_position, so the distance is 1
                distance[next_position] = distance[position] + 1

    return None

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION:
      The following features are considered and combined:
        - Compute the maze distance to the closest food dot
        - Compute the maze distance to the closest capsule
        - If the ghost is scared and close, eat it
        - If the ghost is not scared and close, run away
        - Take into account score (the longer the game is, the lower the score will be)
    """
    "*** YOUR CODE HERE ***"
    infinity = float('inf')
    position = currentGameState.getPacmanPosition()
    score = currentGameState.getScore()
    ghostStates = currentGameState.getGhostStates()
    foodList = currentGameState.getFood().asList()
    capsuleList = currentGameState.getCapsules()

    # if currentGameState.isWin(): return infinity
    # if currentGameState.isLose(): return -infinity

    for ghost in ghostStates:
        d = manhattanDistance(position, ghost.getPosition())
        if ghost.scaredTimer > 6 and d < 2:
            return infinity
        elif ghost.scaredTimer < 5 and d < 2:
            return -infinity

    # Distance to closest food pellet
    # Note that at least one food pellet must exist,
    # otherwise we would have already won!
    cFD = closestItemDistance(currentGameState, foodList)
    if cFD is not None and cFD == 0.0:
        cFD += 1.0
    if cFD is None:
        cFD = 1.0
    foodDistance = 1.0/cFD

    # Distance to closest capsule
    # capsuleDistance = closestItemDistance(currentGameState, capsuleList)
    # capsuleDistance = 0.0 if capsuleDistance is None else 1.0/capsuleDistance
    cCD = closestItemDistance(currentGameState, capsuleList)
    capsuleDistance = 0.0
    if cCD is not None:
        if cCD == 0.0:
            capsuleDistance = 1/(cCD+1)
        else:
            capsuleDistance = 1/cCD

    # Coefficients are kinda arbitrary but this combination seems to work
    return 10.0*foodDistance + 5.0*score + 0.5*capsuleDistance


better = betterEvaluationFunction
