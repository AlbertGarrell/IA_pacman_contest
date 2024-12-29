# baselineTeam.py
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
    def __init__(self, index, time_for_computing = .1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.position_history = []  # To track the last few positions
        self.role_history = []  # Stores recent roles (e.g., ["offensive", "defensive"])


    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)
        self.position_history = []  # Reset position history
        self.role_history = []  # Reset recent roles (e.g., ["offensive", "defensive"])
    
    def get_current_role(self):
        return self.role_history[-1] if self.role_history else None
    
    def detect_loop(self):
        """
        Detects if the agent is in a looping pattern of moves.
        Looks for a repeated position pattern in the last 6 positions.
        """
        if len(self.position_history) < 6:
            return False  # Not enough history to detect a loop
        # Check if the last three positions form a loop
        return (self.position_history[-1] == self.position_history[-3] and
                self.position_history[-2] == self.position_history[-4])

    def assign_role(self, my_dist, other_dist, closest_ghost, enough_score, invaders, we_win, time_left, mid_with):

        if enough_score:
            proposed_role = "defensive"
        
        if time_left < 400 and not we_win:
            proposed_role = "offensive"
        
        if my_dist >= other_dist:
            proposed_role = "offensive"

        else:
            proposed_role = "defensive"

        return proposed_role


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
            #if self.detect_loop() and new_position == self.position_history[-2]:
            #    value -= float('-inf')  # Heavy penalty for perpetuating a loop

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
    
    def invasor_dist(self, game_state):
        """distance closest invader and its state"""
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

        if len(invaders) > 0:
            min_dist = 1000

            for inv in invaders:
                dist = self.get_maze_distance(my_pos, inv.get_position())
                if dist < min_dist:
                    min_dist = dist
                    invader_state = inv
            
            return [dist, invader_state]
        
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
    
    ### CHOOSE ACTION DEPENDING ON AGENT ###
    def choose_action(self, game_state):

        score = self.get_score(game_state)
        time_left = game_state.data.timeleft
        midWidth = game_state.data.layout.width / 2

        team_read_bool = game_state.is_on_red_team(self.index)

        score = score if team_read_bool else -score

        # Adjust score based on team color
        adjusted_score = score if team_read_bool else -score

        # Determine win and score conditions
        we_win = adjusted_score >= 0
        enough_score = adjusted_score > 6

        my_team = self.get_team(game_state)
        my_state = game_state.get_agent_state(self.index)

        my_pos = my_state.get_position()

        for t in my_team:
            state = game_state.get_agent_state(t)

            if t != self.index:
                team_mate = t
                other_pos = state.get_position()
        
        my_dist = abs(self.start[0] - my_pos[0])
        other_dist = abs(self.start[0] - other_pos[0])

        # Computes the distance to the closest ghost

        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        #invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        invaders = [a for a in enemies if a.is_pacman]

        #########################################################################################################################print("Num invaders:", len(invaders))

        ghosts = [a for a in enemies if not a.is_pacman]

        try:
            closest_ghost_dist = self.ghost_dist(game_state)[0]
        except:
            closest_ghost_dist = 5

        """ We compute the position of both our agents in the x axis, and then we compare them:
                - if both agents are in the same x position, they are both offensive agents
                - if one of the agents is further back than the other, it becomes a defensive agent and stays back
                - if the agent that is closest to the opponent's side is eaten, the one defending becomes the 
                  closest one and therefore turns into an offensive agent
        """

        role = self.assign_role(my_dist, other_dist, closest_ghost_dist, enough_score, invaders, we_win, time_left, midWidth)

        ############################################################################################################################print("\tRole: ", role)

        #if (my_dist >= other_dist and closest_ghost >= 7 and not enough_score) or (len(invaders) < 1 and closest_ghost >= 7):
        #if (my_dist >= other_dist and closest_ghost >= 7 and not enough_score):
        if role == 'offensive':
        
            ### OFFENSIVE ###
            actions = game_state.get_legal_actions(self.index)
            my_state = game_state.get_agent_state(self.index)
            my_pos = my_state.get_position()

            food_left = self.get_food(game_state).as_list()

            # Get closest food
            closest_food_dist = 1000
            for food in food_left:
                dist = self.get_maze_distance(my_pos, food)
                if dist < closest_food_dist:
                    closest_food_dist = dist

            carrying = self.get_current_observation().get_agent_state(self.index).num_carrying
            scared_timer = game_state.get_agent_state(self.index).scared_timer

            if self.get_capsules(game_state):  # Check for power capsules first
                #########################################################################################print("Problem = Search Capsule")
                problem = SearchPowerCapsuleProblem(game_state, self, self.index)

            elif closest_food_dist < closest_ghost_dist:  # Prioritize food if it's safe
                #########################################################################################print("Problem = Search Food")
                problem = SearchProblem(game_state, self, self.index)

            elif carrying > 0:  # Return to base if carrying food
                ###########################################################################################print("Problem = Risk Food - Returning to Base")
                problem = ReturnBaseProblem(game_state, self, self.index)

            else:  # Default action
                #########################################################################################print("Problem = Default Action - Searching for Food")
                problem = SearchProblem(game_state, self, self.index)

            return self.aStarSearch(problem, game_state, self.our_heuristic)[0]

        ### DEFENSIVE ###
        actions = game_state.get_legal_actions(self.index)
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Check scared timer (Condition 1)
        scared_timer = my_state.scared_timer
        if scared_timer > 0:  # If scared, go for the opponent's food
            ################################################################################################################print("Defensive agent is scared. Acting offensively.")
            ################################################################################################################print(f"\tScared: {scared_timer}")
            problem = SearchProblem(game_state, self, self.index)
            return self.aStarSearch(problem, game_state, self.our_heuristic)[0]

        # Check for invaders (Condition 2)
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        if len(invaders) > 0:  # If invaders are visible, chase the closest one
            ################################################################################################################print("Invaders detected! Chasing the closest invader.")

            try:
                closest_invader_state = self.invasor_dist(game_state)[1]
                closest_invader_dist = self.invasor_dist(game_state)[0]

                if closest_invader_state:
                    problem = SearchInvaderProblem(game_state, self, self.index, closest_invader_state)
                    return self.aStarSearch(problem, game_state, self.our_heuristic)[0]
            except:
                #############################################################################################################print("Failed to find a path to the invader. Moving to the center of the map.")

                # Calculate the center of the map
                walls = game_state.get_walls()
                map_width = walls.width
                map_height = walls.height
                
                current_position = self.get_position()

                # Calculate the inverted coordinate
                inverted_position = (current_position[0], map_height - current_position[1])

                if abs(current_position[0] - map_width) < 5 or abs(current_position[0] - map_width) > (map_width - 5):
                    # If the agent is near the edge, move to the center
                    inverted_position[0] = map_width // 2

                # Define a PositionSearchProblem to go to the center
                problem = PositionSearchProblem(game_state, start=self.get_position(), goal=inverted_position, walls=walls)

                # Try to move to the center
                actions = self.aStarSearch(problem, game_state, self.our_heuristic)
                if actions:
                    return actions[0]
        
        # Check for disappearing food (Condition 3)
        food_defending = self.get_food_you_are_defending(game_state).as_list()
        previous_food_defending = getattr(self, "previous_food_defending", food_defending)
        self.previous_food_defending = food_defending

        disappeared_food = list(set(previous_food_defending) - set(food_defending))
        if disappeared_food:  # If food disappeared, move to that location
            ############################################################################################################print("Defending disappearing food. Moving to last known location.")
            problem = SearchDisappearedFoodProblem(game_state, self, self.index, disappeared_food)
            actions = self.aStarSearch(problem, game_state, self.our_heuristic)
            if actions:
                return actions[0]
        
        # Default defensive behavior: Oscillate around the center of the map
        ###############################################################################################################print("No immediate threats detected. Patrolling the map center.")

        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

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

        return random.choice(best_actions)
    
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

class SearchDisappearedFoodProblem(PositionSearchProblem):
    """
    A search problem for moving to the location of disappeared food when food is eaten by an invader.
    """

    def __init__(self, gameState, agent, agentIndex, disappeared_food):
        """
        Initialize the problem, focusing on the disappeared food locations.
        """
        self.walls = gameState.get_walls()
        self.startState = gameState.get_agent_state(agentIndex).get_position()
        self.costFn = lambda x: 1  # Uniform cost function
        self.disappeared_food = disappeared_food
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The goal state is reaching the location of the disappeared food.
        """
        return state in self.disappeared_food
    
class SearchInvaderProblem(PositionSearchProblem):
    """
    When we want to hunt an invader (enemy Pacman agents).
    """

    def __init__(self, gameState, agent, agentIndex=0, invaderState=None):
        """
        Initializes the problem to track invaders.
        :param gameState: Current game state
        :param agent: The agent attempting to hunt the invader
        :param agentIndex: Index of the agent in the game
        :param invaderState: The state of the closest invader (optional)
        """
        # Store walls for the PositionSearchProblem setup
        self.walls = gameState.get_walls()
        self.startState = gameState.get_agent_state(agentIndex).get_position()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE
        self.agent = agent
        self.agentIndex = agentIndex

        # Store invader's state if provided
        if invaderState:
            self.goal = invaderState.get_position()
        else:
            # No invaderState provided: look for visible invaders
            self.pos = agent.get_position()
            self.invaders = [
                enemy for enemy in agent.get_opponents(gameState)
                if gameState.get_agent_state(enemy).is_pacman and gameState.get_agent_state(enemy).get_position() is not None
            ]

            if self.invaders:
                # Find the closest invader based on maze distance
                closest_inv = float('inf')
                for inv in self.invaders:
                    inv_pos = gameState.get_agent_state(inv).get_position()
                    dist = agent.get_maze_distance(self.pos, inv_pos)  # Assuming agent has `get_maze_distance`
                    if dist < closest_inv:
                        closest_inv = dist
                        closest_invader = inv

                self.goal = gameState.get_agent_state(closest_invader).get_position()
            else:
                self.goal = None  # No visible invaders

    def isGoalState(self, state):
        """
        Determines whether the given state is the goal state (i.e., the position of the closest invader).
        """
        return state == self.goal


class SearchPatrolFoodProblem(PositionSearchProblem):
    """
    A search problem for patrolling the food you're defending. 
    The goal is to move toward the closest food in your defensive area.
    """

    def __init__(self, gameState, agent, agentIndex, food_defending):
        """
        Initialize the problem, focusing on the food locations you're defending.
        """
        self.walls = gameState.get_walls()
        self.startState = gameState.get_agent_state(agentIndex).get_position()
        self.costFn = lambda x: 1  # Uniform cost function
        self.food_defending = food_defending
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The goal state is reaching the location of the closest food being defended.
        """
        return state in self.food_defending


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

class SpreadOutProblem:
    def __init__(self, game_state, agent, agent_index):
        self.start = game_state.get_agent_position(agent_index)
        self.agent = agent
        self.agent_index = agent_index
        self.teammate_pos = [game_state.get_agent_state(i).get_position()
                             for i in agent.get_team(game_state) if i != agent_index][0]

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        distance_to_teammate = self.agent.get_maze_distance(state, self.teammate_pos)
        return distance_to_teammate >= 3  # Goal: Be at least 3 units apart

    def getSuccessors(self, state):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            next_state = (int(x + dx), int(y + dy))
            if not self.agent.walls[next_state[0]][next_state[1]]:  # Check for walls
                successors.append((next_state, action, 1))  # Cost of 1 for all moves
        return successors
