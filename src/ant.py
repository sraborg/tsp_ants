from __future__ import annotations
from ant_heuristics import AntHeuristic
from typing import Callable, Tuple, TypeVar, TYPE_CHECKING, Optional
import random

# Import following for type checking only
if TYPE_CHECKING:
    _T = TypeVar('_T')
    _KT = TypeVar('_KT')
    _VT = TypeVar('_VT')
    _VT_co = TypeVar('_VT_co')
    from pheromone_graph import PheromoneGraph


class TabuList(dict):

    def __init__(self, num_tabu_neighbors:int = 5, tabu_short_memory:int = 20, tabu_long_memory:int = 100):
        super().__init__()
        self.num_tabu_neighbors = num_tabu_neighbors
        self._tabu_short_memory = tabu_short_memory
        self._tabu_long_memory = tabu_long_memory
        self._expiration = tabu_long_memory - tabu_short_memory

    def __setitem__(self, key, value):
        #print("setItem")
        super().__setitem__(key, value)

    def __getitem__(self, item):
        #print("getItem")
        return super().__getitem__(item)

    def get(self, key: _KT) -> Optional[_VT_co]:
        print("get")
        return super().get(_KT)

    def setdefault(self, __key: _KT, __default: _VT = ...) -> _VT:
        print("setDefault")
        super().setdefault(_KT, _VT)

    def tick(self):
        """ Updates tenure for all solutions in memory and removes solutions that exceed maximum tenure

        """
        for solution in self:
            self[solution]["tenure"] += 1

            # Remove Solutions that have been in memory too long
            if self[solution]["tenure"] > self._tabu_long_memory:
                del solution

    def in_short_term_memory(self, value: Tuple[Tuple[int, int], ...]) -> bool:
        """

        :param value:
        :return:
        """
        return value in self.keys() and self[value]["tenure"] <= self._tabu_short_memory

    def in_long_term_memory(self, value) -> bool:
        """

        :param value:
        :return:
        """
        return value in self.keys() and self[value]["tenure"] > 20

    def get_short_term_memory_keys(self):
        """ Returns a list of all solutions in short term memory

        :return: List of solutions
        """
        keys = [key for key in self.keys() if self.in_short_term_memory(key)]
        return keys

    def get_long_term_memory_keys(self):
        """ Returns a list of all solutions in long term memory

        :return: List of solutions
        """

        keys = [key for key in self.keys() if self.in_long_term_memory(key)]
        return keys

    def add_solution(self, solution):
        """

        :param solution:
        :return:
        """

        # New Solution
        key = solution

        # Case: Not in memory
        if solution not in self:
            self[key] = {}
            self[key]["tenure"] = 0
            self[key]["frequency"] = 1
        # Case: Solution is already in memory
        else:
            self[key]["tenure"] = 0 # reset Tenure
            self[key]["frequency"] += 1

    def update_memory_structure(self, num_tabu_neighbors:int = 5, tabu_short_memory:int = 20, tabu_long_memory:int = 100):
        self.num_tabu_neighbors = num_tabu_neighbors
        self._tabu_short_memory = tabu_short_memory
        self._tabu_long_memory = tabu_long_memory
        #self._expiration = tabu_long_memory - tabu_short_memory


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


class TabuAnt(Ant):

    tabu_list = TabuList()

    def __init__(self, graph: PheromoneGraph, heuristic: AntHeuristic,
                 alpha:int = 1, beta:int = 1, epsilon:int = 1, index:int = None,
                 num_tabu_neighbors:int = 10, tabu_short_memory:int = 20, tabu_long_memory:int = 100, tabu_epsilon=0.05):

        self.index = index
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self._graph = graph
        self._heuristic = heuristic
        self.visited_nodes = []
        self.tour = []
        self.num_neighbors = num_tabu_neighbors
        self.tabu_short_memory = tabu_short_memory
        self.tabu_long_memory = tabu_long_memory
        self.current_solution = None

        # Place Ant on random starting Node
        starting_node = random.choice(tuple(self._graph.nodes))
        self.visited_nodes.append(starting_node)

        super().__init__(graph, heuristic, alpha, beta, epsilon, index)
        self.tabu_list.update_memory_structure(num_tabu_neighbors, tabu_short_memory, tabu_long_memory)
        self.tabu_epsilon = tabu_epsilon

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
        self.current_solution = tuple(self.tour)

        # Tabu Time!
        self.current_solution = self.tabu_search(self.current_solution)

        return self.current_solution

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

    def tabu_neighbor_search(self, solution):

        assert(len(solution) > 1)
        neighbor = list(solution)

        if neighbor is None or len(neighbor) < 2:
            raise ValueError("Schedule must have at least 2 tasks.")

        l = len(solution) - 1
        first_index = random.randint(0, l)
        second_index = random.randint(0, l)

        # Make sure first_index is smallest
        if second_index < first_index:
            temp = first_index
            first_index = second_index
            second_index = temp

        block = solution[second_index:]
        del neighbor[second_index:]
        neighbor[first_index + 1:first_index + 1] = block

        return neighbor

    def tabu_search(self, solution):
        """ Performs Tabu Search, searching neighbors around the local solution

        :param solution: The solution to search around
        :return: The best solution found by tabu search
        """
        # Generate Neighbors
        neighbors = []
        for i in range(self.num_neighbors):
            neighbors.append(self.tabu_neighbor_search(solution))

        # Get Tabu Solutions (e.g. short term memory)
        tabu_solutions = self.tabu_list.get_short_term_memory_keys()

        # Filter out Tabu Solutions (e.g. solutions in short term memory)
        candidate_solutions = list(filter(lambda x: tuple(x) not in tabu_solutions, neighbors))

        # Greedy Selection
        if random.randint(0,1) < self.tabu_epsilon:
            # Check if any neighbor is better
            for candidate in candidate_solutions:
                if self._graph.get_path_value(candidate) >= self._graph.get_path_value(self.current_solution):
                    self.current_solution = tuple(candidate)
        # Probabilistic Selection
        else:
            probabilities = []
            total = 0
            for i, candidate in enumerate(candidate_solutions):
                key = tuple(candidate)
                probabilities.append(self._graph.get_path_value(candidate))

                # Add Frequency values from long term memory
                long_term_memory = self.tabu_list.get_long_term_memory_keys()
                if key in long_term_memory:
                    probabilities[i] /= self.tabu_list[key]["frequency"]        # weight values inversely by frequency

                total += probabilities[i]

            probabilities = [p/total for p in probabilities]
            choice = random.choices(candidate_solutions, probabilities)[0]
            self.current_solution = tuple(choice)

        # Update Tabulist
        self.tabu_list.tick()                                   # Update Tenures
        self.tabu_list.add_solution(self.current_solution)      # Add new value to Tabu List

        return self.current_solution
