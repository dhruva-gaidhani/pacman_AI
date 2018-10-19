# searchAgents.py
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


"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from game import Directions
from game import Agent
from game import Actions
import util
import time
import search

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError, fn + ' is not a search function in search.py.'
        func = getattr(search, fn)
        if 'heuristic' not in func.func_code.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError, heuristic + ' is not a function in searchAgents.py or search.py.'
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError, prob + ' is not a search problem type in SearchAgents.py.'
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception, "No search function provided for SearchAgent"
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
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
            print 'Warning: this does not look like a regular search maze'

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
            dx, dy = Actions.directionToVector(action)
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

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """

    def __init__(self, startingGameState):
        """
        Stores the walls, pacman's starting position and corners.
        """
        self.walls = startingGameState.getWalls()
        self.startingPosition = startingGameState.getPacmanPosition()
        top, right = self.walls.height-2, self.walls.width-2
        self.corners = ((1,1), (1,top), (right, 1), (right, top))
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print 'Warning: no food in corner ' + str(corner)
        self._expanded = 0 # DO NOT CHANGE; Number of search nodes expanded
        # Please add any code here which you would like to use
        # in initializing the problem

        # default cost to use when generating the successors
        self.action_cost = 1

        """
            The below code will be used by cornersHeuristic to calculate the heuristic value for each node.
            The aim of the below code is to find the following details:
                o dpleftbottom[][] -> Store the shortest path from the left bottom (1,1) corner to everypoint in the maze.
                o dprightbottom[][] -> Store the shortest path from the right bottom (right,1) corner to everypoint in the maze.
                o dplefttop[][] -> Store the shortest path from the top left (1,top) corner to everypoint in the maze.
                o dprighttop[][] -> Store the shortest path from the top right (right,top) corner to everypoint in the maze.

        """
        # initiaze our dp arrays with size of (top x right) and default value to be the max value (top x right)
        self.dpleftbottom   = [ [top * right] * (right) for y in range(top)]
        self.dprightbottom  = [ [top * right] * (right) for y in range(top)]
        self.dplefttop      = [ [top * right] * (right) for y in range(top)]
        self.dprighttop     = [ [top * right] * (right) for y in range(top)]

        """
            To populate the DP arrays, we make use of our custom self.generateshortestPath. The function starts from a corner and assign incremental distance
            to its successors. The overall complexity of the method is O(n^2). Thus, it can easily calculate the dp values of maze with sizes upto 2500 x 2500.

            We pass the function, ( corner.x , corner.y , appropriateDP , value of right i.e. max x coordinate , value of top i.e. max y coordinate )

            Also, note that since, graph in maze problem starts from bottom left corner i.e. in a game maze, the bottom left point is (1,1), as compared
            to our DP matrix, where topleft is (1,1), we flip (x,y) when storing in the DP array.
            i.e. point (x,y) in the game maze is (y-1,x-1) in our DP array. ( (y-1),(x-1) => because our graph representation in DP array starts from (0,0)
            and not (1,1) as in a game maze

            i.e. distance of point (x,y) from left bottom (1,1) corner will be dpleftbottom[ y-1 ][ x-1 ]

        """
        self.generateshortestPath(  0       , 0         , self.dpleftbottom  , right , top )
        self.generateshortestPath( top-1    , 0         , self.dplefttop     , right , top )
        self.generateshortestPath( 0        , right - 1 , self.dprightbottom , right , top )
        self.generateshortestPath( top - 1  , right - 1 , self.dprighttop    , right , top )

        """
            Now our DParrays are populated correctly and we have the minimum distance from each point to each of the corners in our DP arrays.
        """

        """
            To optimize the solution a bit, I have also created a dictionary to store the shortest distance between each pair of corners:
            shortestDistanceBetweenCorners => stores the values as ( key,value ) pair,
                    where key -> ( (corner1),(corner2) )

            Between 4 corners, there can be total of 4*3 ordered pair of corners and we need to store distance between each pair of corner in our
            dictionary
        """

        self.shortestDistanceBetweenCorners = {}

        # store the shortest distance between leftbottom and lefttop corner
        # The minimum distance will be min( distance of lefttop from leftbottom , distance of leftbottom from lefttop )
        self.shortestDistanceBetweenCorners[((1, 1), (1, top))] = min(self.dpleftbottom[top - 1][0], self.dplefttop[0][0])
        self.shortestDistanceBetweenCorners[((1, top), (1, 1))] = min(self.dpleftbottom[top - 1][0], self.dplefttop[0][0])

        # Similarly calculate the same for all the other pairs of corners
        self.shortestDistanceBetweenCorners[((1, 1), (right, 1))] = min(self.dpleftbottom[0][right - 1], self.dprightbottom[0][0])
        self.shortestDistanceBetweenCorners[((right, 1), (1, 1))] = min(self.dpleftbottom[0][right - 1], self.dprightbottom[0][0])

        self.shortestDistanceBetweenCorners[((1, 1), (right, top))] = min(self.dpleftbottom[top - 1][right - 1], self.dprighttop[0][0])
        self.shortestDistanceBetweenCorners[((right, top), (1, 1))] = min(self.dpleftbottom[top - 1][right - 1], self.dprighttop[0][0])

        self.shortestDistanceBetweenCorners[((1, top), (right, 1))] = min(self.dplefttop[0][right - 1], self.dprightbottom[top - 1][0])
        self.shortestDistanceBetweenCorners[((right, 1), (1, top))] = min(self.dplefttop[0][right - 1], self.dprightbottom[top - 1][0])

        self.shortestDistanceBetweenCorners[((1, top), (right, top))] = min(self.dplefttop[top - 1][right - 1], self.dprighttop[top - 1][0])
        self.shortestDistanceBetweenCorners[((right, top), (1, top))] = min(self.dplefttop[top - 1][right - 1], self.dprighttop[top - 1][0])

        self.shortestDistanceBetweenCorners[((right, 1), (right, top))] = min(self.dprightbottom[top - 1][right - 1], self.dprighttop[0][right - 1])
        self.shortestDistanceBetweenCorners[((right, top), (right, 1))] = min(self.dprightbottom[top - 1][right - 1], self.dprighttop[0][right - 1])


    def generateshortestPath(self,cornerx,cornery,dp,right,top):

        """
            Find the shortest path from a given corner to each point in the graph:

            Steps:
                1 Start with the given corner, mark it visited, add its distance from the corner as 0.
                    o Also, push the given corner to the queue

                Repeat step 2 to 4 until queue is empty
                    2 Pop a point from the queue.
                    3 Note its distance in our DP matrix.
                    4 Find all its successors
                        4.1 If the successor is not yet visited and is not a wall
                        4.2 Add the successor to the queue and mark its distance as (distance of current-node + 1)

        """

        # initialize our state manager
        state_manager = util.Queue()

        # Set visisted of every node as false
        visited = [[False] * right for k in range(top)]

        # Push the given corner ( (coordinates),distance ) to state manager with distance of 0
        state_manager.push( ((cornerx,cornery),0) )

        # Mark the given corner as visited/expanded
        visited[cornerx][cornery] = True

        while not state_manager.isEmpty():

            # remove the next Node
            state = state_manager.pop()
            # find the coordinate of the current node
            current_node_x,current_node_y = state[0]
            # find the distance of the current node from given corner
            cost_current_node = state[1]
            # Push the distance of the current node to our DP matrix
            dp[current_node_x][current_node_y] = cost_current_node

            for successor in self.getSuccessors(((current_node_y+1,current_node_x+1),self.corners)):

                # find the coordinate of the next node
                next_node_x,next_node_y = successor[0][0]

                # if the next successor node is not visited and is not a wall, add it to the state_manager
                if not visited[next_node_y-1][next_node_x-1]:
                    if self.walls[next_node_x][next_node_y] == False:

                        # push the next_node and its distance to the state_manager
                        state_manager.push(((next_node_y-1, next_node_x-1), cost_current_node + 1))
                        # mark the next_node as visited
                        visited[next_node_y-1][next_node_x-1] = True


    def getStartState(self):
        """
        Returns the start state (in your state space, not the full Pacman state
        space)
        """
        """
            Our state information is now not only the (state,action,cost), but along with this the corners are also part of our state information now.
            So, in addition to returning the startingPostion, we should also return the corner details for the problem.

            Thus, our start state is now a tuple of ( actual game startingPosition, game corners )
        """
        return (self.startingPosition, self.corners)


    def isGoalState(self, state):
        """
        Returns whether this search state is a goal state of the problem.
        """

        # Note our state was of the form ( state position , corners yet to traverse )
        corners_remaining = state[1]

        """
            Our goal state will be the one where corners list is empty i.e. all the corners are already traversed
        """

        # if corners_remaining is not empty return false else return true
        if corners_remaining:
            return False
        else:
            return True


    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
            For a given state, this should return a list of triples, (successor,
            action, stepCost), where 'successor' is a successor to the current
            state, 'action' is the action required to get there, and 'stepCost'
            is the incremental cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            # Add a successor state to the successor list if the action is legal
            # Here's a code snippet for figuring out whether a new position hits a wall:
            #   x,y = currentPosition
            #   dx, dy = Actions.directionToVector(action)
            #   nextx, nexty = int(x + dx), int(y + dy)
            #   hitsWall = self.walls[nextx][nexty]

            x, y = state[0]
            corners_remaining = state[1]
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            hitsWall = self.walls[nextx][nexty]

            # if the next node is a wall, only then it can be a successor where pacman can move
            if not hitsWall :

                next_position = (nextx, nexty)

                # if the next_position is a corner that is not yet traversed, then we should remove this corner from the list of corners
                # to visit after traversing to this node
                corners_left = tuple([ corner for corner in corners_remaining if corner != (nextx, nexty)])

                # push the state,action,cost to the successor list
                successors.append(  ( (next_position,corners_left), action, self.action_cost)  )

        self._expanded += 1 # DO NOT CHANGE
        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999.  This is implemented for you.
        """
        if actions == None: return 999999
        x,y= self.startingPosition
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
        return len(actions)

def cornersHeuristic(state, problem):
    """
    A heuristic for the CornersProblem that you defined.

      state:   The current search state
               (a data structure you chose in your search problem)

      problem: The CornersProblem instance for this layout.

    This function should always return a number that is a lower bound on the
    shortest path from the state to a goal of the problem; i.e.  it should be
    admissible (as well as consistent).
    """
    corners = problem.corners # These are the corner coordinates
    walls = problem.walls # These are the walls of the maze, as a Grid (game.py)

    """
        Shortest path based heuristic function - expands a total of 965 nodes for medium search problem

        Explanation:
            For each unvisited corner:
                o distance_to_corner <= Find the distance from current position to the corner.
                o min_distance_path_from_that_corner_reach_all_other_corners

            eg
            Consider following corners that are unvisited:
                corners:        (1,1),(1,top),(right,1),(right,top)
                current point:  (x,y)

                    lets take corner (1,1) :
                        o find the shortest distance from (x,y) to (1,1) --> note this can be done in O(1) using our problem.dpleftbottom dp matrix
                        o min_distance_path_from_1,1_to_traverse_all_other_corners:
                            these paths can be:
                                (1,1) -> (1,top) -> (right,top) -> (right,1)
                                (1,1) -> (right,top) -> (1,top) -> (right,1)
                                (1,1) -> (right,1) -> (1,top) -> (right,top)
                                (1,1) -> (1,top) -> (right,1) -> (right,top)
                                (1,1) -> (right,top) -> (right,1) -> (1,top)
                                (1,1) -> (right,1) -> (right,top) -> (1,top)

                            find the minimum distance path from the above path.

                            To find all such paths,
                            I have created a generic recursive function,
                            that can enumerate all the paths and select the minimum from them.

                We will have to do the same starting from each of the other corners.

    ****** Note: Due to our pre-calculation using DP matrix, all of the above operations can now happen in constant time ******

    """
    x, y = state[0]
    corner_remaining = state[1]
    top, right = walls.height - 2, walls.width - 2

    # distance array, that will contain
    distance = []
    for (cornerx, cornery) in corner_remaining:

        # left bottom. Enumerate all paths from left bottom corner and find the minimum path
        if cornerx == 1 and cornery == 1:
            distance.append(find_minimum_from_corners(corner_remaining, problem.dpleftbottom, x, cornerx, y, cornery,problem.shortestDistanceBetweenCorners))

        # left top. Enumerate all paths from left top corner and find the minimum path
        if cornerx == 1 and cornery == top:
            distance.append(find_minimum_from_corners(corner_remaining, problem.dplefttop, x, cornerx, y, cornery,problem.shortestDistanceBetweenCorners))

        # right bottom. Enumerate all paths from right bottom corner and find the minimum path
        if cornerx == right and cornery == 1:
            distance.append(find_minimum_from_corners(corner_remaining, problem.dprightbottom, x, cornerx, y, cornery,problem.shortestDistanceBetweenCorners))

        # right top. Enumerate all paths from rght top corner and find the minimum path
        if cornerx == right and cornery == top:
            distance.append(find_minimum_from_corners(corner_remaining, problem.dprighttop, x, cornerx, y, cornery,problem.shortestDistanceBetweenCorners))

    # distance array now contains minimum paths starting from each corner
    # we now sort the distance array and find the ideal corner, from which we will incur a minimum path
    distance.sort()

    # there were actually some unvisited corners pending, then distance array will have at least 1 value and we can now denote the minimum
    # value in the distance array as our heuristic
    if distance:
        return (distance[0])

    # most probably a goal state as no corners left to find the min dist
    return 0

    """ null heuristic - expands a total of 2010522 nodes """
    #return 0 # Default to trivial solution


def find_minimum_from_corners(cor_rem, dp, positionx, cornerx, positiony, cornery, shortestDistancebetweencorners):

    # First add the distance from point(positionx,positiony) in the maze to the corner( cornerx,cornery )
    distance_to_corner = 0
    distance_to_corner += dp[positiony - 1][positionx - 1]

    # mark the corner as visited as we do not want to explore the corner again when we enumerate all the paths below
    visited = {}
    visited[(cornerx, cornery)] = True

    # enumerate all the remaining paths and find the min path distance from them
    minDistanceFound = distMin((cornerx, cornery), cor_rem, visited, shortestDistancebetweencorners)

    # thus mindistance incurred if we choose corner( cornerx,cornery ) is distance_to_corner + minDistanceFound
    return distance_to_corner + minDistanceFound

def distMin( prev,corners,visited,shortestDistance ):

    # initially assign a huge value to minDist
    minDist = 99999999

    # then explore/enumerate for each of the other unvisited corners
    for next_corner in corners:

        # if the next corner is not visited or its visited value is false, then try this path and find the shortest distance from here
        if not ( visited.has_key(next_corner) and visited.get(next_corner) == True ):

            # Mark it as visited, so that it is not selected twice in the path between all corners
            visited[next_corner] = True
            # Find the minimum distance if we select next_corner as our next node in the path
            minDist = min( minDist , shortestDistance[(prev,next_corner)] + distMin(next_corner,corners,visited,shortestDistance) )
            # Again mark it as false, so that other paths can explore this node
            visited[next_corner] = False

    # if minDist is 99999999, then probably there are no more corners to visit. So, return 0
    if minDist == 99999999:
        return 0

    # Else return the minDist
    return minDist

class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem

class FoodSearchProblem:
    """
    A search problem associated with finding the a path that collects all of the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    @profile
    def __init__(self, startingGameState):
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())
        self.walls = startingGameState.getWalls()
        self.startingGameState = startingGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

        # distance_between_food_points <= To store the min distance between each food position
        self.distance_between_food_points = {}

        # foodGrid <= the list of food points
        foodGrid = startingGameState.getFood()

        # mapfoodtonumber <= hashMap to uniquely index each food point. i.e. converts a food-point to a unique index
        self.mapfoodtonumber = {}

        # mapnumbertofood <= reverse hashmap of mapfoodtonumber. i.e. find the food-point given a unique index
        self.mapnumbertofood = {}

        # uniquenumber generator for each food-point
        uniqueNumber = 0

        for food in foodGrid.asList():

            # If food has been assigned a unique number, assign it a unique number and store these details in our hashMap
            if( not self.mapfoodtonumber.has_key(food) ):
                self.mapfoodtonumber[food] = uniqueNumber
                self.mapnumbertofood[uniqueNumber] = food
                uniqueNumber += 1

            for other_food in foodGrid.asList():
                if (not self.mapfoodtonumber.has_key(other_food)):
                    self.mapfoodtonumber[other_food] = uniqueNumber
                    self.mapnumbertofood[uniqueNumber] = other_food
                    uniqueNumber += 1

                # find the distance between food and other-food points
                dist = mazeDistance(food, other_food, startingGameState)

                # store this distance in our DP matrix
                self.distance_between_food_points[ (self.mapfoodtonumber[food],self.mapfoodtonumber[other_food]) ] = dist
                self.distance_between_food_points[ (self.mapfoodtonumber[other_food],self.mapfoodtonumber[food]) ] = dist

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state[1].count() == 0

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors

    def getCostOfActions(self, actions):
        """Returns the cost of a particular sequence of actions.  If those actions
        include an illegal move, return 999999"""
        x,y= self.getStartState()[0]
        cost = 0
        for action in actions:
            # figure out the next state and see whether it's legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]:
                return 999999
            cost += 1
        return cost

class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem

def foodHeuristic(state, problem):
    """
    Your heuristic for the FoodSearchProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come
    up with an admissible heuristic; almost all admissible heuristics will be
    consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the
    other hand, inadmissible or inconsistent heuristics may find optimal
    solutions, so be careful.

    The state is a tuple ( pacmanPosition, foodGrid ) where foodGrid is a Grid
    (see game.py) of either True or False. You can call foodGrid.asList() to get
    a list of food coordinates instead.

    If you want access to info like walls, capsules, etc., you can query the
    problem.  For example, problem.walls gives you a Grid of where the walls
    are.

    If you want to *store* information to be reused in other calls to the
    heuristic, there is a dictionary called problem.heuristicInfo that you can
    use. For example, if you only want to count the walls once and store that
    value, try: problem.heuristicInfo['wallCount'] = problem.walls.count()
    Subsequent calls to this heuristic can access
    problem.heuristicInfo['wallCount']
    """


    """ Credits: https://stackoverflow.com/a/36404229/2802539

        Explanation:
            o Find the distance between 2 farthest food points in the current maze state.
                o Since, in any case, our pacman will always have to travel this distance.
            o Find the distance from curent pacman position to closest of these 2 farthest food points

        To solve the problem, we will have to first find the 2 farthest food points. This can be done in O(n^2) - because we have already pre-calculated the distance
        between each food-point in the maze in the problem class.

        Thus, the heuristic function runs in O(n^2) where, n = number of remaining food-points
    """

    position, foodGrid = state

    # maxDistance <= to store the distance between the 2 farthest points
    maxDistance = -1

    # point1 and point2 <= to store the 2 farthest points
    point1 = None
    point2 = None

    # If there are no food-points left, then we have reached the goal state
    if len(foodGrid.asList())==0:
        return 0

    # calculate the distance between each food-point pair using our DP matrix problem.distance_between_food_points and HashMaps
    for food in foodGrid.asList():
        for other_food in foodGrid.asList():

            # store the details of the 2 farthest points
            if( maxDistance < problem.distance_between_food_points[(problem.mapfoodtonumber[food],problem.mapfoodtonumber[other_food])] ):
                maxDistance = problem.distance_between_food_points[(problem.mapfoodtonumber[food],problem.mapfoodtonumber[other_food])]
                point1 = food
                point2 = other_food

    # now find the minimum distance from current point to point1 or point2
    minDistance = min( mazeDistance(position,point1,problem.startingGameState), mazeDistance(position,point2,problem.startingGameState) )

    # Finally our heuristic is minDistance + maxDistance
    return minDistance + maxDistance

class ClosestDotSearchAgent(SearchAgent):
    "Search for all food using a sequence of searches"
    def registerInitialState(self, state):
        self.actions = []
        currentState = state
        while(currentState.getFood().count() > 0):
            nextPathSegment = self.findPathToClosestDot(currentState) # The missing piece
            self.actions += nextPathSegment
            for action in nextPathSegment:
                legal = currentState.getLegalActions()
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception, 'findPathToClosestDot returned an illegal move: %s!\n%s' % t
                currentState = currentState.generateSuccessor(0, action)
        self.actionIndex = 0
        print 'Path found with cost %d.' % len(self.actions)

    def findPathToClosestDot(self, gameState):
        """
        Returns a path (a list of actions) to the closest dot, starting from
        gameState.
        """
        # Here are some useful elements of the startState
        startPosition = gameState.getPacmanPosition()
        food = gameState.getFood()
        walls = gameState.getWalls()
        problem = AnyFoodSearchProblem(gameState)

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A search problem for finding a path to any food.

    This search problem is just like the PositionSearchProblem, but has a
    different goal test, which you need to fill in below.  The state space and
    successor function do not need to be changed.

    The class definition above, AnyFoodSearchProblem(PositionSearchProblem),
    inherits the methods of the PositionSearchProblem.

    You can use this search problem to help you fill in the findPathToClosestDot
    method.
    """

    def __init__(self, gameState):
        "Stores information from the gameState.  You don't need to change this."
        # Store the food for later reference
        self.food = gameState.getFood()

        # Store info for the PositionSearchProblem (no need to change this)
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        self.costFn = lambda x: 1
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def isGoalState(self, state):
        """
        The state is Pacman's position. Fill this in with a goal test that will
        complete the problem definition.
        """
        x,y = state

        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points, using the search functions
    you have already built. The gameState can be any game state -- Pacman's
    position in that state is ignored.

    Example usage: mazeDistance( (2,4), (5,6), gameState)

    This might be a useful helper function for your ApproximateSearchAgent.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    return len(search.bfs(prob))
