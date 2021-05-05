import argparse
from aco import Aco, TabuAco
from pheromone_graph import PheromoneGraph
# Main Parser
parser = argparse.ArgumentParser()
methods = ["aco", "tabu_aco", "hybrid"]
parser.add_argument('-a', type=int, help='Number of Ants', default=10)
graph = parser.add_mutually_exclusive_group()
graph.add_argument('-n', type=int, help='Number of Cities', default=10)
graph.add_argument('-l', "--load_graph", type=str)
parser.add_argument('-s', '--save_graph', action="store_true")
parser.add_argument('-m', "--method", type=str, choices=methods, required=True)
parser.add_argument('-i', '--iterations', type=int, help='Number of iterations to search for an optimal solution', default=20)
parser.add_argument('--alpha', type=float, help='Alpha parameter used to weight pheromones', default=1.0)
parser.add_argument('--beta', type=float, help='Beta parameter used to weight heurisitc', default=1.0)
parser.add_argument('--epsilon', type=float, help='Probability of being greedy', default=0.05)
parser.add_argument('--max_edge_weight', type=int, help='Probability of being greedy', default=500)

args = parser.parse_args()

# Setup Graph
graph = None
if args.load_graph:
    graph = PheromoneGraph.load_graph(args.load_graph)
else:
    graph = PheromoneGraph.get_pheromone_graph(args.n)
    graph.generate_random_edge_weights(max=args.max_edge_weight)
    if args.save_graph:
        graph.save_graph()

# Setup Algorithm
alg = None
if args.method.upper() == "TABU_ACO":
    alg = TabuAco(graph, args.a, alpha=args.alpha, beta=args.beta, epsilon=args.epsilon, iterations=args.iterations)
elif args.method.upper() == "Hybrid":
    pass
else:
    alg = Aco(graph, args.a, alpha=args.alpha, beta=args.beta, epsilon=args.epsilon, iterations=args.iterations)

# Find Solution
solution = alg.generate_solution()


#alg.draw()

print(solution)
