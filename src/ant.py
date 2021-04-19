from __future__ import annotations
from ant_heuristics import AntHeuristic
from typing import Callable, Tuple, TYPE_CHECKING
import random

# Import following for type checking only
if TYPE_CHECKING:
    from pheromone_graph import PheromoneGraph


class Ant:

    def __init__(self, graph: PheromoneGraph, heuristic: AntHeuristic,
                 alpha:int = 1, beta:int = 1, epsilon:int = 1, index:int = None):

        self.index = index
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self._graph = graph
        self._heuristic = heuristic
        self.visited_nodes = []
        self.tour = []

        # Place Ant on random starting Node
        starting_node = random.choice(tuple(self._graph.nodes))
        self.visited_nodes.append(starting_node)

    def get_tour(self) -> Tuple[Tuple[int, int], ...]:
        """ Generates a tour across the graph

        :return: a tour as a tuple of edges
        """
        while not self._tour_is_complete():
            next_node = self._node_selection()
            self._visit_node(next_node)
            self._graph.local_pheromone_update(self._get_last_edge_taken())

        # Complete Cycle by adding final edge backing to starting node
        start_node = self.visited_nodes[0]
        last_node = self.visited_nodes[-1]
        final_edge = (last_node, start_node)
        self.tour.append(final_edge)

        return tuple(self.tour)

    def _visit_node(self, node: int):
        """ Has the ant visit the node

        :param node: The node to visit
        """

        # Make sure the node is actually part of the graph
        assert(node < self._graph.number_of_nodes())

        current_node = self.current_node()

        # Update visited_nodes
        self.visited_nodes.append(node)

        # Add Edge to tour
        edge = (current_node, node)
        self.tour.append(edge)

    def _node_selection(self) -> int:
        """ Method for selecting the next node to visit

        :return: the next node visit
        """

        current_node = self.current_node()

        # Determine which nodes remain to be visited
        candidate_nodes: tuple[int, ...] = self._get_candidate_nodes()

        # Make a greedy Selection
        if self.epsilon < random.uniform(0,1):

            # Choose the node that gives the best immediate value based of the heuristic & pheromone level
            choices = list(candidate_nodes)
            choices.sort(key=lambda x: self._graph[current_node][x]['pheromone'] ** self.alpha *\
                         self._heuristic(self._graph, current_node, candidate) ** self.beta,
                         reverse=True)
            return choices[0]

        # Make weighted probabilistic selection
        else:
            # Calculate the probability of selecting each candidate node
            probabilities = []
            for candidate in candidate_nodes:
                prob = self._graph[current_node][candidate]['pheromone'] ** self.alpha *\
                       self._heuristic(self._graph, current_node, candidate) ** self.beta
                probabilities.append(prob)

            total = sum(probabilities)
            normalized_probabilities = [p/total for p in probabilities]

            # Select the next node
            selected_node = random.choices(candidate_nodes, normalized_probabilities)[0]

            return selected_node

    def _tour_is_complete(self) -> bool:
        """ Checks whether a tour is complete or not be checking if the number of visited
        matches the number of nodes in the graph

        :return: True/False
        """
        return len(self.visited_nodes) == len(self._graph.nodes)

    def _get_last_edge_taken(self) -> Tuple[int, int]:
        """ Simple helper function to get the ant's last edge taken

        :return: The last edge taken (e.g. a tuple of two nodes)
        """
        return self.tour[-1]

    def current_node(self) -> int:
        """ Simple helper function to get the ant's current location

        :return: The current node the ant is at
        """
        return self.visited_nodes[-1]

    def _get_candidate_nodes(self) -> tuple[int, ...]:
        """ Determines which nodes can be visited from the current node that haven't been visited yet

        :return: a tuple containing the candidate nodes
        """
        current_node = self.current_node()
        neighbors = self._graph.neighbors(current_node)
        return tuple([node for node in neighbors if node not in self.visited_nodes])


class TabuAnt:

    tabu_list = []

    def __init__(self, graph: PheromoneGraph, heuristic: AntHeuristic,
                 alpha:int = 1, beta:int = 1, epsilon:int = 1, index:int = None):

        self.index = index
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self._graph = graph
        self._heuristic = heuristic
        self.visited_nodes = []
        self.tour = []

        # Place Ant on random starting Node
        starting_node = random.choice(tuple(self._graph.nodes))
        self.visited_nodes.append(starting_node)

    def get_tour(self) -> Tuple[Tuple[int, int], ...]:
        """ Generates a tour across the graph

        :return: a tour as a tuple of edges
        """
        while not self._tour_is_complete():
            next_node = self._node_selection()
            self._visit_node(next_node)
            self._graph.local_pheromone_update(self._get_last_edge_taken())

        # Complete Cycle by adding final edge backing to starting node
        start_node = self.visited_nodes[0]
        last_node = self.visited_nodes[-1]
        final_edge = (last_node, start_node)
        self.tour.append(final_edge)

        # Tabu

        return tuple(self.tour)

    def _visit_node(self, node: int):
        """ Has the ant visit the node

        :param node: The node to visit
        """

        # Make sure the node is actually part of the graph
        assert(node < self._graph.number_of_nodes())

        current_node = self.current_node()

        # Update visited_nodes
        self.visited_nodes.append(node)

        # Add Edge to tour
        edge = (current_node, node)
        self.tour.append(edge)

    def _node_selection(self) -> int:
        """ Method for selecting the next node to visit

        :return: the next node visit
        """

        current_node = self.current_node()

        # Determine which nodes remain to be visited
        candidate_nodes: tuple[int, ...] = self._get_candidate_nodes()

        # Make a greedy Selection
        if self.epsilon < random.uniform(0,1):

            # Choose the node that gives the best immediate value based of the heuristic & pheromone level
            choices = list(candidate_nodes)
            choices.sort(key=lambda x: self._graph[current_node][x]['pheromone'] ** self.alpha *\
                         self._heuristic(self._graph, current_node, candidate) ** self.beta,
                         reverse=True)
            return choices[0]

        # Make weighted probabilistic selection
        else:
            # Calculate the probability of selecting each candidate node
            probabilities = []
            for candidate in candidate_nodes:
                prob = self._graph[current_node][candidate]['pheromone'] ** self.alpha *\
                       self._heuristic(self._graph, current_node, candidate) ** self.beta
                probabilities.append(prob)

            total = sum(probabilities)
            normalized_probabilities = [p/total for p in probabilities]

            # Select the next node
            selected_node = random.choices(candidate_nodes, normalized_probabilities)[0]

            return selected_node

    def _tour_is_complete(self) -> bool:
        """ Checks whether a tour is complete or not be checking if the number of visited
        matches the number of nodes in the graph

        :return: True/False
        """
        return len(self.visited_nodes) == len(self._graph.nodes)

    def _get_last_edge_taken(self) -> Tuple[int, int]:
        """ Simple helper function to get the ant's last edge taken

        :return: The last edge taken (e.g. a tuple of two nodes)
        """
        return self.tour[-1]

    def current_node(self) -> int:
        """ Simple helper function to get the ant's current location

        :return: The current node the ant is at
        """
        return self.visited_nodes[-1]

    def _get_candidate_nodes(self) -> tuple[int, ...]:
        """ Determines which nodes can be visited from the current node that haven't been visited yet

        :return: a tuple containing the candidate nodes
        """
        current_node = self.current_node()
        neighbors = self._graph.neighbors(current_node)
        return tuple([node for node in neighbors if node not in self.visited_nodes])