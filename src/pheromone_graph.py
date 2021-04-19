from __future__ import annotations
import networkx as nx
from multiprocessing import Process
import matplotlib.pyplot as plt
import random
from typing import Tuple, TYPE_CHECKING
import types

# Import Following for type checking only
if TYPE_CHECKING:
    from ant import Ant


class PheromoneGraph(nx.classes.Graph):

    def __init(self):
        raise RuntimeError("Use Factory Method instead of calling constructor")

    @staticmethod
    def get_pheromone_graph(num_cities: int = 10, local_evaporation_rate:float = 0.1,
                            global_evaporation_rate:float = 0.2, initial_pheromone_value:float = 1):
        """ Creates an instance of a networkx graph. Adds two new methods to the instance at runtime

        :param local_evaporation_rate:
        :param num_cities: the number of cities
        :return: a fully connected graph
        """

        # Generate the Fully Connected Graph
        g = nx.complete_graph(num_cities)

        # add additional properties to the graph
        g.initial_pheromone_value = initial_pheromone_value
        g.local_evaporation_rate = local_evaporation_rate
        g.global_evaporation_rate = global_evaporation_rate


        # Bind the Additional Methods
        g.initialize_pheromone_levels = types.MethodType(initialize_pheromone_levels, g)
        g.pheromone_update = types.MethodType(pheromone_update, g)
        g.local_pheromone_update = types.MethodType(local_pheromone_update, g)
        g.generate_random_edge_weights = types.MethodType(generate_random_edge_weights, g)
        g.get_path_value = types.MethodType(get_path_value, g)
        g.draw = types.MethodType(draw, g)
        #g._draw = types.MethodType(_draw, g)
        #g.close_drawing = types.MethodType(close_drawing, g)

        # Initialize Pheromones
        g.initialize_pheromone_levels()

        g.pos = nx.spring_layout(g)  # positions for all nodes
        plt.figure(3, figsize=(12, 12))
        plt.ion()
        plt.show()
        g.draw()

        return g


##
#
# The Following are methods that are bound at runtime to the networkx graph class
#
##

def generate_random_edge_weights(self, min=1, max=100):
    """ Addes random weight values to each edge. The range of the values are bound by the min/max arguments

    :param self: Reference to the graph object
    :param min: Minimum weight value
    :param max: Maximum weight value
    """
    for n1, n2 in self.edges:
        self.edges[n1,n2]['weight'] = random.randint(min, max)


def initialize_pheromone_levels(self):
    """ Initializes the pheromone levels for all edges in the graph

    :param self: Reference to the graph
    """
    for n1, n2 in self.edges:
        self.edges[n1,n2]['pheromone'] = self.initial_pheromone_value


def pheromone_update(self, ant: Ant):
    """ Pheromone update method to be potentially used after each batch of ants

    :param self: Reference to the graph
    :param ant:  The ant during the update
    """
    # Update the pheromone levels for each edge the ant took
    for n1, n2 in ant.get_tour():
        self.edges[n1, n2]['pheromone'] = (1 - self.global_evaporation_rate) * self.edges[n1, n2]['pheromone'] + \
                                          self.global_evaporation_rate * self.initial_pheromone_value


def local_pheromone_update(self, edge: Tuple[int, int]):
    """ Pheromone update method to be potentially used by each ant during each iteration

    :param edge: The edge whose pheromone levels need to be updated
    :param self: Reference to the graph
    :param ant:  The ant during the update
    """

    # Update the pheromone levels for each edge the ant took
    n1 = edge[0]
    n2 = edge[1]
    self.edges[n1, n2]['pheromone'] = (1 - self.local_evaporation_rate) *self.edges[n1, n2]['pheromone'] +\
                                      self.local_evaporation_rate * self.initial_pheromone_value


def get_path_value(self, path: Tuple[Tuple[int,int], ...]) -> float:
    """ Calculates the value for the path (e.g. sums up the weights for the edges in the path)

    :param self: reference to graph object
    :param path: the path (list of edges) to evaluate
    :return: the value
    """
    value = sum([self.edges[n1, n2]['weight'] for n1, n2 in path])
    return value


def draw(self, edge_highlights: Tuple[Tuple[int,int], ...] = (), best_tour:Tuple[Tuple[int,int], ...] = (),
         highlight_color: str = 'b', best_tour_color: str = 'r', block=False, iteration=0, ant=0, best_ant:int = -1,
         print_tour:bool = False):
    """

    :param self: Reference to the graph
    :param edge_highlights: A path/tour to highlight
    :param best_tour: Another tour to highlight
    :param highlight_color: color used for the path/tour to highlight
    :param best_tour_color: Color used for
    :param block: Flag for blocking execution (used to keep the final graph displayed)
    :param iteration: Algorithm iteration
    :param ant: index of current ant searching
    :param best_ant: index of best ant
    :param print_tour: Flag used to determine whether to display tour list
    """

    # Graph Setup
    plt.clf()
    plt.axis("off")

    # Update & Display Title
    title = "Iteration: " + str(iteration) + " | Ant: " + str(ant)
    if best_ant >= 0:
        value = self.get_path_value(best_tour)
        title = title + " | Best Path Value: " + str(value)
        if print_tour:
            order = [edge[0] for edge in best_tour]
            order.append(order[0])
            title = title + "\n Best Path: "
            title = title + str(order)
    plt.title(title)

    # nodes
    nx.draw_networkx_nodes(self, self.pos, node_size=700)

    # edges
    edges = [(u, v) for (u, v, d) in self.edges(data=True) if (u, v) not in edge_highlights]
    nx.draw_networkx_edges(self, self.pos, edgelist=edges, width=1, alpha=0.25)
    nx.draw_networkx_edges(self, self.pos, edgelist=edge_highlights, width=1, alpha=1, edge_color=highlight_color)
    nx.draw_networkx_edges(self, self.pos, edgelist=best_tour, width=1, alpha=1, edge_color=best_tour_color)

    # labels
    nx.draw_networkx_labels(self, self.pos, font_size=15, font_family="sans-serif", font_color="w")
    edge_labels = nx.get_edge_attributes(self, 'weight')
    nx.draw_networkx_edge_labels(self, self.pos, label_pos=0.5, edge_labels=edge_labels)

    # Determines whether to block execution (e.g. pause execution & keep the current drawing showing)
    if block:
        plt.ioff()
        plt.show()
    else:
        plt.draw()
        plt.pause(0.1)




def close_drawing(self):
    plt.close("all")