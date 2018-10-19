# search.py
# ---------
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
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    """
    """
        We can make use of our generic search algorithm, where we need to design our state manager as a data structure that can pop the last element pushed
        to mimic the idea of DFS.

        Thus, we can make use of Util.Stack data structure, which is based on LIFO principles

    """

    return generic_search(problem, util.Stack())
    # util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    """
        We can make use of our generic search algorithm, where we need to design our state manager as a data structure that can pop the first element that was
        pushed into it, to mimic the idea of BFS.

        Thus, we can make use of Util.Queue data structure, which is based on FIFO principles

    """
    return generic_search(problem, util.Queue())
    # util.raiseNotDefined()


def generic_search(problem, state_call_manager):
    """
        Defines a general algorithm to search a graph.
        Parameters are structure, which can be any data structure with .push() and .pop() methods, and problem, which is the
        search problem.
        """

    # Visited dictionary to store the expanded nodes
    visited = {}

    # state space manager.
    # We will push all the states and their metadata to this data structure.
    # Depending on the type of the data structure (Stack,Queue,etc), the search function will process accordingly.
    # We are going to push ( path_to_the_node, state_of_node , cost_to_the_node )
    #                           where, state_of_node <= ( node_location,action,cost )
    state_call_manager.push(((), problem.getStartState(), 0))

    # iterate over the state space manager to retrieve the stored states
    while not state_call_manager.isEmpty():

        # current_state <= ( path_to_the_node, state_of_node , cost_to_the_node )
        current_state = state_call_manager.pop()

        # current_path_directions <= ( path_to_the_node ) i.e. it is a tuple of states ( node_location,action,cost )
        current_path_directions = current_state[0]
        # current_node <= ( state_of_node )
        current_node = current_state[1]
        # cost_till_now <= ( cost_to_the_node )
        cost_till_now = current_state[2]


        # If the current node is the goal state, then extract each state from the current_path_directions
        # and create an array of actions to take the path and return the array, which is our solution to the
        # search problem
        if problem.isGoalState(current_node):
            # state <= ( node_location,action,cost )
            return [state[1] for state in current_path_directions]

        # To avoid exploring the same node again, add a check if the current node has already been expanded.
        # If yes, we should move on to the next state in our state space manager
        if visited.has_key(current_node):
            continue

        # Mark the current node as visited/expanded
        visited[current_node] = True

        # Look-up each of the successor from the current_node and add them to the state manager if it has not been yet explored
        for successor in problem.getSuccessors(current_node):
            # successor <= ( node_location,action,cost )

            # next_node <= coordinate of the next node
            next_node = successor[0]
            # cost_next_node <= cost till visiting the current node + cost from current_node to next_node (successor[2])
            cost_next_node = cost_till_now + successor[2]

            # Add next_node to the state manager only of it has not been explored
            if not visited.has_key(next_node):
                # we need to add ( path_to_the_node, state_of_node , cost_to_the_node ) to the state manager

                # path_to_next_node <= path to current node + path from current-node to next-node
                path_to_next_node = current_path_directions + (successor,)

                # push the successor metadata to the state manager
                state_call_manager.push((tuple(path_to_next_node), next_node, cost_next_node))

    # Goal State not found.
    return False


def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    """
        We can again make use of our generic search algorithm, except that in case of UCS, our data structure will be a priority queue.
        Since, we need to select the lowest cost node in each iteration, we need a min-heap data structure.

        Util.PriorityQueue is a min-heap. But, it requires us to explicitly push the priority in the push function. Thus, it won't be useful
        as we have already made use of a generic push function in our general search function.

        To avoid re-writing the entire logic again, we can make use of Util.PriorityQueueWithFunction, where we can pass a function reference to
        calculate the priority for each element, whenever we push an element to the heap.

        I have used a anonymous lambda function, which takes in the data-structure element and returns the priority.

        Note: our data-structure element is ( path_to_the_node, state_of_node , cost_to_the_node ), in our general search algorithm. So our anonymous
        function should return cost_to_the_node, i.e. the last element of the tuple.
    """
    return generic_search(problem, util.PriorityQueueWithFunction(lambda (x, y, z): z))
    # util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    """
        Idea here is similar to our generic search solution, except that in this case, our cost_to_the_path should also incorporate the heuristic information.
        Thus, our state manager will now need to include the heuristic information for each node pushed to the state manager.
    """

    # Visited dictionary to store the expanded nodes
    visited = {}

    # state space manager.
    # We will push all the states and their metadata to this data structure.
    # We are going to push ( path_to_the_node , state_of_node , cost_to_the_node_along_with_heuristic , heuristic_of_the_current_node )
    #                           where, state_of_node <= ( node_location,action,cost )

    # we make use of util.PriorityQueueWithFunction, and pass the lambda function reference, that will return the 3rd value in the tuple
    # ( path_to_the_node , state_of_node , cost_to_the_node_along_with_heuristic , heuristic_of_the_current_node ) i.e. cost_to_the_node_along_with_heuristic
    state_call_manager = util.PriorityQueueWithFunction(lambda (p, q, r, s): r)

    # calculate the heuristic of the start node, using the heuristic function passed to our method
    heuristic_start_node = heuristic(problem.getStartState(), problem)

    # Push the start node to the state manager
    state_call_manager.push(((), problem.getStartState(), 0 + heuristic_start_node, heuristic_start_node))  # (path,currentNode,cost,heuristic)

    # iterate over the state space manager to retrieve the stored states
    while not state_call_manager.isEmpty():

        # current_state <= ( path_to_the_node, state_of_node , cost_to_the_node_along_with_heuristic , heuristic_of_the_current_node )
        current_state = state_call_manager.pop()
        # current_path_directions <= path_to_the_node
        current_path_directions = current_state[0]
        # current_node <= state_of_node
        current_node = current_state[1]
        # heuristic_current_node <= heuristic_of_the_current_node
        heuristic_current_node = current_state[3];
        # cost_till_now <= cost_to_the_current_node_with_heuristic - heuristic_of_current_node
        cost_till_now = current_state[2] - heuristic_current_node

        # If the current node is the goal state, then extract each state from the current_path_directions
        # and create an array of actions to take the path and return the array, which is our solution to the
        # search problem
        if problem.isGoalState(current_node):
            # state <= ( node_location,action,cost )
            return [state[1] for state in current_path_directions]

        # To avoid exploring the same node again, add a check if the current node has already been expanded.
        # If yes, we should move on to the next state in our state space manager
        if visited.has_key(current_node):
            continue

        # Mark the current node as visited/expanded
        visited[current_node] = True

        # Look-up each of the successor from the current_node and add them to the state manager if it has not been yet explored
        for successor in problem.getSuccessors(current_node):
            # successor <= ( node_location,action,cost )

            # next_node <= coordinate of the next node
            next_node = successor[0]

            # heuristic_next_node <= calculate the heuristic of the next_node using the heuristic function passed to our method.
            heuristic_next_node = heuristic(next_node, problem)

            # cost_next_node <= cost till visiting the current node + cost from current_node to next_node (successor[2]) + heuristic_next_node
            cost_next_node = cost_till_now + successor[2] + heuristic_next_node

            if not visited.has_key(next_node):
                # we need to add ( path_to_the_node , state_of_node , cost_to_the_node_along_with_heuristic , heuristic_of_the_current_node )
                # to the state manager

                # path_to_next_node <= path to current node + path from current-node to next-node
                path_to_next_node = current_path_directions + (successor,)

                # push the successor metadata to the state manager
                state_call_manager.push((tuple(path_to_next_node), next_node, cost_next_node, heuristic_next_node))

    # Goal State not found.
    return False


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
