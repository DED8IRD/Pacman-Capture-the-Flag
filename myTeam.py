# myTeam.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import random, time, util
from util import nearestPoint
from game import Directions, Actions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAgent', second = 'DefensiveAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.
  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########
class AStarAgent(CaptureAgent):
  "A search agent using A*"
 
  ##########
  # Search #
  ##########
  def createSearchProblem(self, startingGameState):
    agent = self

    class SearchProblem:
      """
      A search problem associated with finding the a path that collects all of the 
      food on the enemy side in a capture the flag game while avoiding enemy ghosts.
      
      A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
        pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
        foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food 
      """
      def __init__(self, startingGameState):
        
        self.start = (startingGameState.getAgentState(agent.index).getPosition(), agent.getFood(startingGameState))
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0
        self.heuristicInfo = {} # A dictionary for the heuristic to store information
          
      def startingState(self):
        return self.start
      
      def isGoal(self, state):
        pos, food = state
        return food.count() <= 2

      def successorStates(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        print "state", state
        pos, food = state
        succ = []
        self._expanded += 1
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
          x,y = pos
          dx, dy = Actions.directionToVector(direction)
          nextx, nexty = int(x + dx), int(y + dy)
          if not self.walls[nextx][nexty]:
            nextFood = food.copy()
            nextFood[nextx][nexty] = False
            succ.append( ( ((nextx, nexty), nextFood), direction, 1) )
        # print succ
        return succ

      def actionsCost(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.startingState()[0]
        cost = 0
        for action in actions:
          # figure out the next state and see whether it's legal
          dx, dy = Actions.directionToVector(action)
          x, y = int(x + dx), int(y + dy)
          if self.walls[x][y]:
            return 999999
          cost += 1
        return cost
    return SearchProblem(startingGameState)

  def registerInitialState(self, gameState):
    self.start = (gameState.getAgentState(self.index).getPosition(), self.getFood(gameState))
    # self.food = self.getFood(gameState)
    # print self.food
    self.walls = gameState.getWalls()
    self.startingGameState = gameState
    self._expanded = 0
    self.heuristicInfo = {}

    self.starttime = time.time()
    self.problem = self.createSearchProblem(gameState)
    self.searchFunction = lambda prob: search.aStarSearch(prob, self.heuristic)
    self.actions = self.aStarSearch(self.problem, self.heuristic(gameState, self.problem))

  def chooseAction(self, gameState):
    """
    Returns the next action in the path chosen earlier (in registerInitialState).  Return
    Directions.STOP if there is no further action to take.
    
    state: a GameState object (pacman.py)
    """
    if 'actionIndex' not in dir(self): self.actionIndex = 0
    i = self.actionIndex
    self.actionIndex += 1
    if i < len(self.actions):
      return self.actions[i]    
    else:
      return Directions.STOP

  def heuristic(self, gameState, problem):
    pos, food = problem.startingState()
    foodRemaining = food.count()

    return foodRemaining    

  def aStarSearch(self, problem, heuristic):
    fringe = util.PriorityQueue()
    start = problem.startingState()
    explored = set()
    # root = (start, [])
    print "root = ", start[0]
    print "problem = ", self.problem
    print "heurstic = ", self.heuristic(start, problem)
    fringe.push( (start, [], 0), self.heuristic(start, problem) )   # Push root onto fringe

    while not fringe.isEmpty():
      node, actions = fringe.pop()                     # Get new state
      print "node: \n",node, "\n",actions
      if problem.isGoal(node):                         # Goal test
        return actions

      explored.add(node)
      nextState = problem.successorStates(node)        # Expand node
      print "nextState = ", nextState
      for coord, nextAction, cost in nextState:
        if node not in explored:                       # Path checking
          # coord, nextAction, cost = successor
          # print "successor = ", successor
          path = actions + [nextAction]
          next = coord, path, cost
          cost = problem.actionsCost(path) + self.heuristic(next, problem)
          fringe.push(next, cost)
          print fringe
          print "fringe push :", next, cost
    return []
    

class SmartAgent(CaptureAgent):
  """
  A base class for search agents that chooses score-maximizing actions.
  """

  def registerInitialState(self, gameState):

      CaptureAgent.registerInitialState(self, gameState)
      self.boundary_top = True
      if gameState.getAgentState(self.index).getPosition()[0] == 1:
          self.isRed = False
      else:
          self.isRed = True

      self.boundaries = self.boundaryTravel(gameState)
      self.treeDepth = 3

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [action for action, value in zip(actions, values) if value == maxValue]

    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

  def boundaryTravel(self, gameState):
      return (0, 0), (0, 0)


# class OffensiveAgent(SmartAgent):
#   """
#   A reflex agent that seeks food. This is an agent
#   we give you to get an idea of what an offensive agent might look like,
#   but it is by no means the best or only way to build an offensive agent.
#   """

#   def getAction(self, gameState):
#     """
#     Returns the expectimax action using self.depth and self.evaluationFunction
#     All ghosts should be modeled as choosing uniformly at random from their
#     legal moves.
#     """
#     opponents = {}
#     for enemy in self.getOpponents(gameState):
#       opponents[enemy] = gameState.getAgentState(enemy).getPosition()
#     directions = {'north': (0, 1), 'south': (0, -1), 'east': (1, 0), 'west': (-1, 0)}
#     ghost_weights = {'distance': 5, 'scared': 5}

#     # def expectation(opponents):

#     def get_ghost_actions(current_pos):
#       walls = gameState.getWalls().asList()

#       max_x = max([wall[0] for wall in walls])
#       max_y = max([wall[1] for wall in walls])

#       actions = []
#       for direction in directions:
#         action = directions[direction]
#         new_pos = (int(current_pos[0] + action[0]), int(current_pos[1] + action[1]))
#         if new_pos not in walls:
#           if (1 <= new_pos[0] < max_x) and (1 <= new_pos[1] < max_y):
#             actions.append(direction.title())
#       return actions

#     def get_new_position(current_pos, action):
#       act = directions[[direction for direction in directions if str(action).lower() == direction][0]]
#       return (current_pos[0] + act[0], current_pos[1] + act[1])

#     def ghost_eval(gamestate, opponents, opponent):
#       newPos = opponents[opponent]
#       enemy = gamestate.getAgentState(opponent)
#       myPos = gamestate.getAgentState(self.index).getPosition()

#       if enemy.scaredTimer != 0:
#         distance = -self.getMazeDistance(myPos, newPos)*ghost_weights['distance']
#       else:
#         distance = self.getMazeDistance(myPos, newPos)*ghost_weights['distance']

#       return distance

#     def minimax(gamestate, depth, agent, opponents, alpha=-float('inf'), beta=float('inf')):

#       # Get legal moves per agent
#       legalActions = [action for action in gamestate.getLegalActions(self.index) if action != Directions.STOP]

#       # Generate optimal action recursively
#       actions = {}
#       if agent == self.index:
#         maxVal = -float('inf')
#         for action in legalActions:
#           eval = self.evaluate(gamestate, action)
#           if depth == self.treeDepth:
#             value = eval
#           else:
#             value = eval+minimax(self.getSuccessor(gamestate, action), depth, agent+1, opponents, alpha, beta)
#           maxVal = max(maxVal, value)
#           if beta < maxVal:
#             return maxVal
#           else:
#             alpha = max(alpha, maxVal)
#           if depth == 1:
#             actions[value] = action
#         if depth == 1:          # If you're up to the first depth, return a legal action
#           return actions[maxVal]
#         return maxVal
#       else:
#         minVal = float('inf')
#         for opponent in opponents:
#           if gamestate.getAgentState(opponent).getPosition() is not None:
#             legalActions = get_ghost_actions(opponents[opponent])
#             for action in legalActions:
#               new_opponents = opponents.copy()
#               new_opponents[opponent] = get_new_position(opponents[opponent], action)
#               ghost_val = ghost_eval(gamestate, new_opponents, opponent)
#               value = ghost_val + minimax(gamestate, depth+1, self.index, new_opponents, alpha, beta)
#               minVal = min(minVal, value)
#               if minVal < alpha:
#                 return minVal
#               else:
#                 beta = min(beta, minVal)
#         if minVal == float('inf'):
#           return 0
#         return minVal

#     return minimax(gameState, 1, self.index, opponents)


#   def getFeatures(self, gameState, action):
#     features = util.Counter()
#     successor = self.getSuccessor(gameState, action)
#     features['successorScore'] = self.getScore(successor)

#     myState = successor.getAgentState(self.index)
#     myPos = myState.getPosition()    

#     # Computes distance to enemy ghosts
#     enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
#     ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
#     invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

#     features['invaderDistance'] = 0.0
#     if len(invaders) > 0:
#         features['invaderDistance'] = min([self.getMazeDistance(myPos, invader.getPosition()) for invader in invaders]) + 1

#     # features['numGhosts'] = len(ghosts)
#     if len(ghosts) > 0:
#       ghostEval = 0.0
#       for ghost in ghosts:
#         ghostDistance = self.getMazeDistance( myPos, ghost.getPosition() ) 
#         if ghost.scaredTimer == 0:       # If ghost is not scared
#           if ghostDistance <= 1:         # If your agent touches a ghost,
#             ghostEval = -float('inf')    # the ghostDistance feature evaluates to -infinity
#             break
#           else:
#             if ghostDistance < abs(ghostEval):
#               ghostEval = ghostDistance
#         else:   # If ghost is scared
#           if ghostDistance == 0:
#             ghostEval = ghostEval+100
#             break
#           else:
#             if ghostDistance < abs(ghostEval):
#               ghostEval = -ghostDistance
#       features['distanceToGhost'] = ghostEval

#     # Compute distance to the nearest food
#     foodList = self.getFood(successor).asList()
#     if len(foodList) > 0: # This should always be True,  but better safe than sorry
#       minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
#       features['distanceToFood'] = minDistance
#       features['foodRemaining'] = len(foodList)

#     # Compute distance to capsules
#     capsules = self.getCapsules(successor)
#     if len(capsules) > 0:
#       minDistance = min([ self.getMazeDistance(myPos, capsule) for capsule in capsules ])
#       if minDistance == 0: minDistance = -100
#       features['distanceToCapsules'] = minDistance

#     if action == Directions.STOP: features['stop'] = 1
#     rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
#     if action == rev: features['reverse'] = 1

#     return features

#   def getWeights(self, gameState, action):
#     return {'successorScore': 100, 'invaderDistance': -50, 'distanceToFood': -1, 'foodRemaining': -1, 'distanceToGhost': 3, 'distanceToCapsules': -1, 'stop': -50, 'ghostScared': 50, 'reverse': -5}



class OffensiveAgent(SmartAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()    

    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]

    # Computes distance to enemy ghosts
    if len(invaders) > 0:
        features['invaderDistance'] = min([self.getMazeDistance(myPos, invader.getPosition()) for invader in invaders]) + 1

    if len(ghosts) > 0:
      ghostEval = 0.0
      ghostScared = 0.0
      regGhosts = [ghost for ghost in ghosts if ghost.scaredTimer == 0]
      scaredGhosts = [ghost for ghost in ghosts if ghost.scaredTimer > 0]
      if len(regGhosts) > 0: 
        ghostEval = min([self.getMazeDistance(myPos, ghost.getPosition()) for ghost in regGhosts])
        if ghostEval == 0:  ghostEval = -float('inf')
         
      if len(scaredGhosts) > 0: 
        ghostScared = min([self.getMazeDistance(myPos, ghost.getPosition()) for ghost in scaredGhosts])
      if ghostScared < ghostEval or ghostEval == 0:
        if ghostScared == 0: ghostScared = -50
        features['ghostScared'] = ghostScared
      features['distanceToGhost'] = ghostEval


    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      distance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = distance
      features['foodRemaining'] = len(foodList)

    # Compute distance to capsules
    capsules = self.getCapsules(successor)
    if len(capsules) > 0:
      minDistance = min([ self.getMazeDistance(myPos, capsule) for capsule in capsules ])
      features['distanceToCapsules'] = minDistance

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'invaderDistance': -50, 'distanceToFood': -3, 'foodRemaining': -1, 'distanceToGhost': 50, 'ghostScared': -1, 'distanceToCapsules': -1, 'stop': -100, 'reverse': -50}


class DefensiveAgent(SmartAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    numInvaders = len([invader for invader in enemies if invader.isPacman])
    features['numInvaders'] = numInvaders
    defenseFood = self.getFoodYouAreDefending(successor).asList()
    numFood = len([food for food in defenseFood])
    boundaries = self.boundaries

    # Computes whether we're on defense (1) or offense (0)
    defense = 10
    if myState.isPacman: features['onDefense'] = 0
    else: features['onDefense'] = 1

    # Incentivizes returning to defense if defense food below 10 or enemy Pacmen are invading our side
    if numFood < 5 or numInvaders > 0:  
      # features['onDefense'] = defense * numInvaders^2
      distance = max([self.getMazeDistance(myPos, food) for food in defenseFood])
      if self.isRed:
        boundX = max([food[0] for food in defenseFood])
        if myPos[0] < boundX: features['distanceToFood'] = distance
      else:
        boundX = min([food[0] for food in defenseFood])
        if myPos[0] > boundX: features['distanceToFood'] = distance     

    # If the agent needs to go to the upper bound, the bound is set to the upper bound. Otherwise it's the lower bound
    if self.boundary_top is True: bound = boundaries[0]
    else: bound = boundaries[1]

    # If the agent has reached the upper bound, set the top boundary to false and vice versa
    if myPos == bound: self.boundary_top = not(self.boundary_top)
    features['bound'] = self.getMazeDistance(myPos, bound)

    # Computes distance to invaders we can see and their distance to the food we are defending
    # if len(invaders) == 0:
    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance + 1

    dist = 0.0
    distances = [self.getMazeDistance(myPos, enemy.getPosition()) for enemy in enemies if (enemy.scaredTimer != 0 and enemy.getPosition() is not None)]
    if len(distances) > 0:
      dist = min(distances) + 1
    features['invaderDistance'] = dist
    features['defenseFoodDistance'] = 0.
    # else:
    if len(invaders) > 0:
      distances = [self.getMazeDistance(myPos, enemy.getPosition()) for enemy in enemies if (enemy.scaredTimer == 0 and enemy.getPosition() is not None)]
      if len(distances) > 0:
        features['enemyChase'] = min(distances) + 1

      features['invaderDistance'] = min([self.getMazeDistance(myPos, invader.getPosition()) for invader in invaders]) + 1
      features['defenseFoodDistance'] = min([min([self.getMazeDistance(invader.getPosition(), food) for invader in invaders]) for food in defenseFood]) + 1
      features['distanceToFood'] = 0.0


    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 50, 'invaderDistance': -100, 'distanceToFood': -1, 'defenseFoodDistance': -8, 'enemyChase': 10, 'bound': -10, 'stop': -100, 'reverse': -50, }


  def boundaryTravel(self, gameState):
    """
    Returns two points that act as a boundary line along which the agent travels
    """
    defenseFood = self.getFoodYouAreDefending(gameState).asList()
    walls = gameState.getWalls().asList()
    max_y = max([food[1] for food in defenseFood])
    max_wall = max([wall[1] for wall in walls])

    if not self.isRed:
        mid_x = max([food[0] for food in defenseFood])
    else:
        mid_x = min([food[0] for food in defenseFood])


    # lower bound is 1/3 of grid. Upper bound is 2/3 of grid
    lower = max_y/3
    upper = (max_y*2)/3

    # If the positions are illegal states, add 1 to get a legal state
    while (mid_x, lower) in walls: lower += 1
    if lower >= max_wall:
      lower = max_y/3
      while (mid_x, lower) in walls: 
        if self.isRed: mid_x += 1
        else: mid_x -= 1
    while (mid_x, upper) in walls: upper += 1
    if upper >= max_wall:
      upper = max_y*2/3
      while (mid_x, upper) in walls: 
        if self.isRed: mid_x += 1
        else: mid_x -= 1
    return (mid_x, lower), (mid_x, upper)
