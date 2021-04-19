from pheromone_graph import PheromoneGraph
from ant import Ant, TabuAnt
import ant_heuristics
import typing as typ


class Aco:

    def __init__(self, graph: PheromoneGraph, num_ants:int = 20, heuristic: typ.Callable = lambda x: 1,
                 alpha:int = 1, beta:int = 1, epsilon:int = 1, iterations:int = 100):

        self._graph = graph
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.num_ants = num_ants
        self.heuristic = heuristic
        self.iterations = iterations

    def generate_solution(self):
        """ Searches for the best tour through the graph

        :return: Returns the best tour found
        """
        best_solution = None
        best_ant:int = None
        for i in range(self.iterations):

            # Generate Ants
            for j in range(self.num_ants):
                ant = self.generate_ant()
                ant.index = j
                tour = ant.get_tour()

                # Update Best Path found (smaller is better)
                if best_solution is None or self._graph.get_path_value(tour) < self._graph.get_path_value(best_solution):
                    best_solution = tour
                    best_ant = ant

                # Draw Current Ant Path
                self._graph.draw(ant.get_tour(), best_tour=best_solution, iteration=i, ant=j, best_ant=best_ant.index)

            if best_ant is not None:
                self._graph.pheromone_update(ant)

            self._graph.draw(ant.get_tour(), best_tour=best_solution, iteration=i, ant=best_ant.index, best_ant= best_ant.index)

        self._graph.draw(best_tour=best_solution, iteration=i, ant=best_ant.index, best_ant= best_ant.index, block=True, print_tour=True)
        return best_solution

    def draw(self):
        self._graph.draw()

    def generate_ant(self):
        return Ant(self._graph, ant_heuristics.NearestNeighbor())


class TabuAco(Aco):

    def generate_ant(self):
        return TabuAnt(self._graph, ant_heuristics.NearestNeighbor())