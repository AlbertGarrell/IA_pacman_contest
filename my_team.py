# myTeam.py
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


# myTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# ############
# Authors:
# Albert Garrell Golobardes
# Arol Garcia Rodríguez
# ############

import random
import util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point
from contest.game import Actions
from contest.capture import GameState

max_num_food = 20

#################
# Shared Assignments #
#################

# This dictionary will store which agent is chasing which enemy.
# Tracking which agent is chasing which invader
chase_status = {}


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

    def assign_role(self, my_dist, other_dist, adjusted_score, enough_score, invaders, we_win, time_left, mid_with):

        if time_left > 1050:
            return "offensive"
        
        if adjusted_score < -7:
            return "offensive"

        if enough_score:
            return "defensive"
        
        if time_left < 400 and not we_win:
            return "offensive"
        
        if my_dist >= other_dist:
            return "offensive"
        else:
            return "defensive"

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
        
        return [None, None]

class OffensiveDefensive(ReflexCaptureAgent):
    """
    A specialized agent that switches between offensive and defensive roles dynamically.
    The switching strategy prioritizes efficient transitions when one agent is removed (e.g., eaten).
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
        we_win = adjusted_score > 0
        enough_score = adjusted_score > 5

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

        #print("Num invaders:", len(invaders))


        try:
            closest_ghost_dist = self.ghost_dist(game_state)[0]
        except:
            closest_ghost_dist = 5


        role = self.assign_role(my_dist, other_dist, adjusted_score, enough_score, invaders, we_win, time_left, midWidth)

        ### OFFENSIVE ###
        if role == 'offensive':  
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

            if carrying >= 8:
                # print(f"{carrying} food collected. Returning to base.")
                problem = ReturnBaseProblem(game_state, self, self.index)

            elif self.get_capsules(game_state):  # Check for power capsules first
                # print("Problem = Search Capsule")
                problem = SearchPowerCapsuleProblem(game_state, self, self.index)

            elif carrying > 0:  # Return to base if carrying food
                max_food_carry = 2  # Máxima comida permitida si estamos ganando
                safety_margin = 8  # Margen de seguridad para regresar a tiempo

                # Calcula la distancia más corta al punto más cercano de la frontera
                closest_boundary_dist = min(
                    [self.get_maze_distance(my_pos, boundary) for boundary in self.boundary(game_state)]
                )
                
                # Condición para regresar
                if time_left < closest_boundary_dist*5 + safety_margin:
                    # print(f"Time Constraint: Returning to base. Time left: {time_left}, Dist to base: {closest_boundary_dist}")
                    problem = ReturnBaseProblem(game_state, self, self.index)
                #elif we_win and carrying >= max_food_carry and not nearby_food:
                elif (closest_ghost_dist <= 3):
                    # print(f"Too close to ghost. Returning Home")
                    problem = ReturnBaseProblem(game_state, self, self.index)
                elif (closest_food_dist < closest_ghost_dist + 4) and (closest_ghost_dist>3):  # Prioritize food if it's safe
                    ########################################################################################## print("Problem = Search Food")
                    # print(f"Secure to search food.")
                    problem = SearchProblem(game_state, self, self.index)
                elif we_win and carrying >= max_food_carry:
                    # print(f"Winning and carrying {carrying} food. Returning to base.")
                    problem = ReturnBaseProblem(game_state, self, self.index)
                else:
                    problem = SearchProblem(game_state, self, self.index)

                # Llama a aStarSearch y maneja resultados vacíos
                actions = self.aStarSearch(problem, game_state, self.our_heuristic)
                if not actions:  # Si no se encontró un camino
                    # print("No valid actions found with A* search, choosing random action.")
                    actions = game_state.get_legal_actions(self.index)
                    return random.choice(actions)  # Acción aleatoria como fallback
                return actions[0]

            else:  # Default action
                ########################################################################################## print("Problem = Default Action - Searching for Food")
                problem = SearchProblem(game_state, self, self.index)

            actions = self.aStarSearch(problem, game_state, self.our_heuristic)
            if not actions:  # Si no se encuentra un camino válido
                # print(f"No actions found with A* for problem: {type(problem).__name__}")
                actions = game_state.get_legal_actions(self.index)
                return random.choice(actions)  # Acción aleatoria como fallback
            return actions[0]

        ### DEFENSIVE ###
        if role == 'defensive':
            # 0. In case we were attacking with food and now we are defensive
            carrying = self.get_current_observation().get_agent_state(self.index).num_carrying
            if carrying > 0:  # Return to base if carrying food
                # print(f"Returning to base with {carrying} food.")
                problem = ReturnBaseProblem(game_state, self, self.index)
            # 1. Check scared timer
            scared_timer = my_state.scared_timer
            if scared_timer > 0 and time_left > 500:  # If scared, go for the opponent's food
                # print(f"Agent {self.index}: Scared timer active, switching to offensive.")
                problem = SearchProblem(game_state, self, self.index)
                return self.aStarSearch(problem, game_state, self.our_heuristic)[0]

            # 2. Check for the closest invader
            closest_invader_state = self.invasor_dist(game_state)[1]
            closest_invader_dist = self.invasor_dist(game_state)[0]

            if closest_invader_dist is not None:
                # Find the invader's index
                for opponent_index in self.get_opponents(game_state):
                    opponent_state = game_state.get_agent_state(opponent_index)
                    if opponent_state == closest_invader_state:  # Match AgentState
                        invader_id = opponent_index
                        break
                else:
                    # print(f"Agent {self.index}: No matching opponent found for closest invader.")
                    invader_id = None

                if invader_id is not None:
                    # Check if another agent is already chasing this invader
                    if invader_id in chase_status:
                        if chase_status[invader_id] != self.index and chase_status[invader_id] != 0:
                            # Another agent is already chasing
                            # print(f"Agent {self.index}: Another agent is chasing invader {invader_id}.")
                            pass
                            
                        else:
                            # Continue chasing if already assigned
                            # print(f"Agent {self.index}: Continuing to chase invader {invader_id}.")
                            problem = SearchInvaderProblem(game_state, self, self.index, closest_invader_state)
                            actions = self.aStarSearch(problem, game_state, self.our_heuristic)
                            if actions:
                                return actions[0]
                    else:
                        # Assign this agent to chase the invader
                        chase_status[invader_id] = self.index
                        # print(f"Agent {self.index} is now chasing invader {invader_id}.")
                        problem = SearchInvaderProblem(game_state, self, self.index, closest_invader_state)
                        actions = self.aStarSearch(problem, game_state, self.our_heuristic)
                        if actions:
                            return actions[0]

            # 3. Clear chase status if no invaders remain or agent stops chasing
            for invader_id in list(chase_status.keys()):
                if chase_status[invader_id] == self.index:  # Only clear entries for this agent
                    invader_state = game_state.get_agent_state(invader_id)
                    if not invader_state.is_pacman or invader_state.get_position() is None:
                        del chase_status[invader_id]
                        # print(f"Agent {self.index}: Stopped chasing invader {invader_id} (no longer visible).")

            # 4. Check for disappearing food
            previous_game_state = self.get_previous_observation()

            if not hasattr(self, "last_disappeared_food"):
                self.last_disappeared_food = []

            if previous_game_state:
                current_food = self.get_food_you_are_defending(game_state).as_list()
                previous_food = self.get_food_you_are_defending(previous_game_state).as_list()
                disappeared_food = list(set(previous_food) - set(current_food))

                if disappeared_food:
                    self.last_disappeared_food = disappeared_food

            if self.last_disappeared_food:
                problem = SearchDisappearedFoodProblem(game_state, self, self.index, self.last_disappeared_food)
                actions = self.aStarSearch(problem, game_state, self.our_heuristic)
                if actions:
                    next_action = actions[0]
                    next_position = game_state.generate_successor(self.index, next_action).get_agent_position(self.index)

                    if next_position in self.last_disappeared_food:
                        self.last_disappeared_food.remove(next_position)

                    return next_action

            # 5. Default defensive behavior
            actions = game_state.get_legal_actions(self.index)
            values = [self.evaluate(game_state, a) for a in actions]
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
