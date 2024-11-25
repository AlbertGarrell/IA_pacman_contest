# prova.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point
from contest.game import Actions
from contest.capture import GameState

#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first = 'OffensiveDefensive', second = 'OffensiveDefensive', num_training = 0):
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
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.position_history = []  # To track the last few positions

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.position_history = []  # Reset position history

    def detect_loop(self):  
        """
        Detects if the agent is in a looping pattern of 2 or 3 moves.
        """
        if len(self.position_history) < 6:
            return False  # Not enough history to detect 3-move loops

        # Check for a 2-move loop (A <-> B)
        if (self.position_history[-1] == self.position_history[-3] and
            self.position_history[-2] == self.position_history[-4]):
            return True

        # Check for a 3-move loop (A -> B -> C -> A)
        if (self.position_history[-1] == self.position_history[-4] and
            self.position_history[-2] == self.position_history[-5] and
            self.position_history[-3] == self.position_history[-6]):
            return True

        return False



    def choose_action(self, game_state):
        """
        Picks among the actions with the highest evaluation, avoiding loops where possible.
        """
        actions = game_state.get_legal_actions(self.index)
        best_action = None
        max_value = float('-inf')

        # Evaluate all actions
        for action in actions:
            successor = self.get_successor(game_state, action)
            new_position = successor.get_agent_position(self.index)

            # Evaluate the action
            value = self.evaluate(game_state, action)

            # Penalize moves that perpetuate a detected loop
            if self.detect_loop() and new_position == self.position_history[-2]:
                value -= 500  # Heavy penalty for perpetuating a loop

            # Penalize STOP action, but less severely
            if action == Directions.STOP:
                value -= 100

            if value > max_value:
                max_value = value
                best_action = action

        # If no valid action is found, choose randomly
        if not best_action:
            best_action = random.choice(actions)

        # Update position history
        self.position_history.append(game_state.get_agent_position(self.index))
        if len(self.position_history) > 6:  # Limit history size
            self.position_history.pop(0)

        return best_action

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor
    
    def boundary(self, game_state):
        """function to obtain the list of points in the boundary"""
        midWidth = game_state.data.layout.width / 2
        height =  game_state.data.layout.height
        boudaries = []
        validPos = []

        if self.red:
            i = midWidth - 1
        else:
            i = midWidth + 1

        for j in range(height):
            boudaries.append((i, j))

        for i in boudaries:
            if not game_state.has_wall(int(i[0]),int(i[1])):
                validPos.append(i)

        return validPos

    def ghost_dist(self, game_state):
        """distance closest ghost and its state"""
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman and a.get_position() is not None]

        if len(ghosts) > 0:
            min_dist = 1000

            for ghost in ghosts:
                dist = self.get_maze_distance(my_pos, ghost.get_position())

                if dist < min_dist:
                    min_dist = dist
                    ghost_state = ghost

            return [dist, ghost_state]
         
        return None

class OffensiveDefensive(ReflexCaptureAgent):
    """
    Our strategy is to have one agent defending and one attacking, but alternating them so 
    that the one that is closest to the opponent's side of the grid is the attacker. This way, 
    if our offensive agent is eaten, the deffensive one automatically becommes an attacker, and 
    the new agent that comes out of our start point is a defensive agent. With this, we avoid losing 
    the time that it takes for the new ghost to get to the opponent's side and since the new one 
    starts in our side, it can just start defending it.
    This switch is computed when we choose the best action.
    We have modified the offensive agent with an A* strategy following the concepts in lab1.
    """
    ### OFFENSIVE ###
    def nullHeuristic(state, problem = None):
        return 0
    
    # heuristic to avoid ghosts
    def our_heuristic(self, state, game_state):
        heuristic_value = 0

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        ghosts = [a for a in enemies if not a.is_pacman]

        # distance to the nearest ghost
        if len(ghosts) > 0:
            ghosts_pos = [ghost.get_position() for ghost in ghosts]
            ghosts_dists = []

            if len(ghosts_pos) > 0:
                for gp in ghosts_pos:
                    if gp != None:
                        ghosts_dists.append(self.get_maze_distance(state, gp))
            
                if len(ghosts_dists) > 0:
                    nearest_ghost = min(ghosts_dists)
                    scared_timer = game_state.get_agent_state(self.index).scared_timer

                    if nearest_ghost < 5:
                        if scared_timer == 0:
                            heuristic_value = heuristic_value + 100 * (5 - nearest_ghost)
                        else:
                            heuristic_value = heuristic_value - 75 * (5 - nearest_ghost)
    
        return heuristic_value

    # following our lab1 submission
    def aStarSearch(self, problem, game_state, heuristic = nullHeuristic):
        """Search the node that has the lowest combined cost and heuristic first."""
        expanded = [] # list of states already visited
        aStarQueue = util.PriorityQueue() # using priority queue 
        initialState = problem.getStartState()

        # priority queue with (state, path list, cost)
        aStarQueue.push((initialState, [], 0), heuristic(initialState, game_state))

        while not aStarQueue.is_empty():
            state, path, cost = aStarQueue.pop()

            # if the state visited is the goal, return the list of moves
            if problem.isGoalState(state):
                return path 

            # visit all successors
            if state not in expanded:
                for s, d, c in problem.getSuccessors(state):
                    new_path = path + [d]
                    new_cost = cost + c
                    h_value = heuristic(s, game_state)
                    # Calculate the priority (cost + heuristic)
                    priority = new_cost + h_value
                    aStarQueue.push((s, new_path, new_cost), priority)

                # Update the visited states list
                expanded.append(state)

        return []
    
    ### DEFENSIVE ###
    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman: features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights
    
def choose_action(self, game_state):
    score = self.get_score(game_state)
    team_read_bool = game_state.is_on_red_team(self.index)

    # Adjust score based on team color
    adjusted_score = score if team_read_bool else -score

    # Determine win and score conditions
    we_win = adjusted_score > 0
    enough_score = adjusted_score > 4

    my_team = self.get_team(game_state)
    my_state = game_state.get_agent_state(self.index)
    my_pos = my_state.get_position()

    # Determine positions of teammates
    for t in my_team:
        state = game_state.get_agent_state(t)
        if t != self.index:
            other_pos = state.get_position()

    my_dist = abs(self.start[0] - my_pos[0])
    other_dist = abs(self.start[0] - other_pos[0])

    enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
    invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
    ghosts = [a for a in enemies if not a.is_pacman]

    try:
        closest_ghost = self.ghost_dist(game_state)[0]
    except:
        closest_ghost = 5


    ### Offensive Behavior ###
    if (my_dist >= other_dist and closest_ghost >= 5 and not enough_score) or (len(invaders) < 1 and closest_ghost >= 5):
        actions = game_state.get_legal_actions(self.index)
        food_left = len(self.get_food(game_state).as_list())
        carrying = self.get_current_observation().get_agent_state(self.index).num_carrying
        scared_timer = game_state.get_agent_state(self.index).scared_timer

        closest_ghost_dist = None
        if self.ghost_dist(game_state):
            closest_ghost_dist = self.ghost_dist(game_state)[0]

        if carrying == 0 and food_left == 0:
            return 'Stop'

        # Choose the best action using offensive logic
        best_action = None
        max_value = float('-inf')
        for action in actions:
            successor = self.get_successor(game_state, action)
            new_position = successor.get_agent_position(self.index)

            value = self.evaluate(game_state, action)

            # Penalize actions that perpetuate detected loops
            if self.detect_loop() and (new_position == self.position_history[-2] or
                                  new_position == self.position_history[-3]):
                value -= 5000  # Penalize loops heavily 

            if value > max_value:
                max_value = value
                best_action = action

        # Fallback to a random action if no valid action is found
        if not best_action:
            best_action = random.choice(actions)

        # Update position history for loop detection
        self.position_history.append(my_pos)
        if len(self.position_history) > 6:
            self.position_history.pop(0)

        return best_action

    ### Defensive Behavior ###
    actions = game_state.get_legal_actions(self.index)

    # Evaluate all legal actions
    values = []
    for action in actions:
        successor = self.get_successor(game_state, action)
        new_position = successor.get_agent_position(self.index)

        # Evaluate the action
        value = self.evaluate(game_state, action)

        # Penalize actions that perpetuate detected loops
        if self.detect_loop() and (new_position == self.position_history[-2] or
                              new_position == self.position_history[-3]):
            value -= 5000  # Large penalty for repeating loops

        values.append(value)

    # Select the best action(s)
    max_value = max(values)
    best_actions = [a for a, v in zip(actions, values) if v == max_value]

    # Special case: Handle situations with minimal food left
    food_left = len(self.get_food(game_state).as_list())
    if food_left <= 2:
        best_dist = 9999
        best_action = None
        for action in actions:
            successor = self.get_successor(game_state, action)
            pos2 = successor.get_agent_position(self.index)
            dist = self.get_maze_distance(self.start, pos2)
            if dist < best_dist:
                best_action = action
                best_dist = dist
        return best_action

    # Fallback to a random choice if needed
    selected_action = random.choice(best_actions)

    # Update position history for loop detection
    self.position_history.append(game_state.get_agent_position(self.index))
    if len(self.position_history) > 6:
        self.position_history.pop(0)

    return selected_action

    
# from PositionSearchProblem of lab1 (searchAgents.py)
class PositionSearchProblem:
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal = (1,1), start = None, warn = True, visualize = True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.direction_to_vector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

# from AnyFoodSearchProblem of lab1 (searchAgents.py)
class SearchProblem(PositionSearchProblem):

    def __init__(self, gameState, agent, agentIndex = 0):
        """Stores information from the gameState.  You don't need to change this."""
        # Store the food for later reference
        self.food = agent.get_food(gameState)

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.get_walls()
        self.startState = gameState.get_agent_state(agentIndex).get_position()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

        # added 
        self.capsule = agent.get_capsules(gameState)
        self.carry = gameState.get_agent_state(agentIndex).num_carrying
        self.foodLeft = len(self.food.as_list())

    def isGoalState(self, state):
    
        if state in self.food.as_list():
            return True
            
        return False
    
class ReturnBaseProblem(PositionSearchProblem):
    """when the goal is to return to base"""

    def __init__(self, gameState, agent, agentIndex = 0):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = agent.get_food(gameState)

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.get_walls()
        self.startState = gameState.get_agent_state(agentIndex).get_position()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

        # added 
        self.capsule = agent.get_capsules(gameState)
        self.carry = gameState.get_agent_state(agentIndex).num_carrying
        self.foodLeft = len(self.food.as_list())
        self.boundaryPos = agent.boundary(gameState)
    
    def isGoalState(self, state):
        return state in self.boundaryPos

class SearchPowerCapsuleProblem(PositionSearchProblem):
    """when the goal is to eat the Power Capsule"""

    def __init__(self, gameState, agent, agentIndex = 0):
        """Stores information from the gameState.  You don't need to change this."""
        # Store the food for later reference
        self.food = agent.get_food(gameState)

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.get_walls()
        self.startState = gameState.get_agent_state(agentIndex).get_position()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

        # added 
        self.capsule = agent.get_capsules(gameState)
        self.carry = gameState.get_agent_state(agentIndex).num_carrying
        self.foodLeft = len(self.food.as_list())


    def isGoalState(self, state):
        return state in self.capsule # the goal state is the location of capsule

class SearchGhostProblem(PositionSearchProblem):
    """when the agent has eatean a Power Capsule and the goal is to eat the ghosts if they are close"""

    def __init__(self, gameState, agent, agentIndex = 0):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = agent.get_food(gameState)

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.get_walls()
        self.startState = gameState.get_agent_state(agentIndex).get_position()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

        # added 
        self.capsule = agent.get_capsules(gameState)
        self.carry = gameState.get_agent_state(agentIndex).num_carrying
        self.foodLeft = len(self.food.as_list())
        self.ghost_dist = agent.ghost_dist(gameState)[1] 


    def isGoalState(self, state):
         # the goal state is the location of the closestghost
        return state == self.ghost_dist.get_position()