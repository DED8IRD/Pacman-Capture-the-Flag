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
from game import Directions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveReflexAgent', second = 'DefensiveReflexAgent'):
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

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on). 
    
    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    ''' 
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py. 
    '''
    CaptureAgent.registerInitialState(self, gameState)

    ''' 
    Your initialization code goes here, if you need any.
    '''


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    ''' 
    You should change this in your own agent.
    '''

    return random.choice(actions)

class SmartAgent(CaptureAgent):
  """
  A base class for search agents that chooses score-maximizing actions.
  """
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)
    values = [self.evaluate(gameState, a) for a in actions]

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


class OffensiveReflexAgent(SmartAgent):
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

    # Computes distance to enemy ghosts
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    if len(ghosts) > 0:
      scaredGhosts = [ghost for ghost in ghosts if ghost.scaredTimer > 0]
      regGhosts = [ghost for ghost in ghosts if ghost.scaredTimer == 0]
      ghostDistance = 0.0
      ghostEval = 0.0
      for ghost in scaredGhosts:       # If ghost is scared
        ghostDistance = self.getMazeDistance( myPos, ghost.getPosition() ) 
        if ghostDistance >= 1:
          ghostEval = 100
          break
        else:
          if ghostDistance < abs(ghostEval):
            ghostEval = 10 - ghostDistance   

      for ghost in regGhosts:          # If ghost is not scared
        ghostDist = self.getMazeDistance( myPos, ghost.getPosition() ) 
        if ghostDist <= 1:             # If your agent touches a ghost,
          ghostEval = -1000            # the ghostDistance feature evaluates to -1000
          break
        else:
          if ghostDist < ghostDistance:
              ghostEval = ghostDist
      features['distanceToGhost'] = ghostEval

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
      # features['foodRemaining'] = len(foodList)

    # Compute distance to capsules
    capsules = self.getCapsules(successor)
    if len(capsules) > 0:
      minDistance = min([ self.getMazeDistance(myPos, capsule) for capsule in capsules ])
      features['distanceToCapsules'] = minDistance
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1, 'foodRemaining': -1, 'distanceToGhost': 1, 'distanceToCapsules': -1}


class DefensiveReflexAgent(SmartAgent):
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

    # Computes distance to invaders we can see and their distance to the food we are defending
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    defenseFood = self.getFoodYouAreDefending(successor).asList()
    features['numInvaders'] = len(invaders)

    # Computes whether we're on defense (1) or offense (0)
    defense = 10
    if myState.isPacman: features['onDefense'] = 0
    else: features['onDefense'] = 1
    numInvaders = len([invader for invader in enemies if invader.isPacman])
    if numInvaders > 0:
      features['onDefense'] = defense * numInvaders
      if myState.isPacman: features['onDefense'] = - numInvaders * defense

    # Evaluates to -infinity if there are 2 or less food remaining on your side
    numFood = len([food for food in defenseFood])
    features['foodRemaining'] = 15.0 / numFood
    if numFood < 15:
      features['onDefense'] = 100 - numFood

    if numInvaders == 0:  # If no invaders, go on the offensive
      # Compute distance to the nearest food
      foodList = self.getFood(successor).asList()
      if len(foodList) > 0: # This should always be True,  but better safe than sorry
        minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
        features['distanceToFood'] = minDistance

      dist = 0.0
      enemyDists = [ self.getMazeDistance(myPos, enemy.getPosition()) for enemy in enemies if (enemy.scaredTimer == 0 and enemy.getPosition() is not None) ]
      if len(enemyDists) > 0:
        dist = min(enemyDists)
      else:             # If agent gets eaten by ghost
        dist = 10       # Evaluates to -100
      features['invaderDistance'] = dist

      scaredDists = [ self.getMazeDistance(myPos, enemy.getPosition()) for enemy in enemies if (enemy.scaredTimer > 0 and enemy.getPosition() is not None) ]
      if len(scaredDists) > 0:
        dist = min(scaredDists)
        if dist >= 1:   # Evaluates to +50 if agent eats scared ghost
          features['scaredDistance'] = 50

    else: # If there are invaders, exhibit defensive behavior
      if len(invaders) > 0:
        # Calculate agent's distance to invader
        dist = [ self.getMazeDistance( myPos, invader.getPosition() ) for invader in invaders ]
        nearestInvader = min(dist)
        if myState.scaredTimer == 0:  # This agent is not scared
          if nearestInvader >= 1:     # If defensive agent eats ghost
            nearestInvader == -10     # Evaluates to +100
          features['invaderDistance'] = nearestInvader 
          features['defenseFoodDistance'] = min([min([self.getMazeDistance(invader.getPosition(), food) for invader in invaders]) for food in defenseFood])
        else:   # This agent is scared
          features['invaderDistance'] = - nearestInvader
    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 5, 'foodRemaining': -1, 'invaderDistance': -10, 'scaredDistance': 1, 'distanceToFood': -1, 'defenseFoodDistance': -8, 'stop': -100, 'reverse': -50, 'enemyChase': 10}
