from abc import ABC, abstractmethod
from typing import Callable


class AntHeuristic(ABC, Callable):

    @abstractmethod
    def __call__(self, graph, current_node, candidate_nodes):
        pass


class NeutralHeuristic(AntHeuristic):

    def __call__(self, graph, current_node, candidate):
        """ Always returns a constant value of 1 regardless of arguments received

        :param graph: Reference to the graph
        :param current_node: The current Node
        :param candidate: The candidate Node
        :return: 1
        """
        return 1


class NearestNeighbor(AntHeuristic):

    def __call__(self, graph, current_node, candidate):
        """ Returns a value equal to the reciprical of the edge weight between the current and candidate nodes

            :param graph: Reference to the graph
            :param current_node: The current Node
            :param candidate: The candidate Node
            :return: Always returns a constant value of 1
        """
        value = 1 / graph[current_node][candidate]["weight"]
        return value