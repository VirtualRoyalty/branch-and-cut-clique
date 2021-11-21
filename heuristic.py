import numpy as np
import networkx as nx
from utils import read_graph_file
from benchmarks import EASY, MEDIUM, HARD, REST
from problem import ProblemHandler


class HeuristicMaxClique:

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.strategies = [
            self.largest_first_randomized,
            self.color_first_randomized
        ]
        self.coloring_strategies = ProblemHandler.STRATEGIES

    def run(self):
        best_clique_size = 0
        best_clique = None
        for strategy in self.strategies:
            for coloring_strategy in self.coloring_strategies:
                found_clique = strategy(self.graph, strategy=coloring_strategy)
                if best_clique_size < len(found_clique):
                    best_clique_size = len(found_clique)
                    best_clique = found_clique
        # return result in a form [0, 1, 0..., 0]
        # return [1.0 if i + 1 in best_clique else 0.0
        #         for i in range(self.graph.number_of_nodes())]
        return [1.0 if i in best_clique else 0.0
                for i in range(self.graph.number_of_nodes())]

    @staticmethod
    def color_first_randomized(graph: nx.Graph, n_iterations: int = 7,
                               strategy=nx.coloring.strategy_random_sequential) -> set:
        best_clique = set()
        best_clique_size = 0
        for _ in range(n_iterations):
            coloring_dct = nx.coloring.greedy_color(graph, strategy=strategy)
            sorted_coloring = sorted(coloring_dct.items(), key=lambda item: item[1], reverse=True)
            nodes = [node for node, color in sorted_coloring]
            clique = set()
            first_index = 0
            while len(nodes) > 0:
                neighbors = list(graph.neighbors(nodes[first_index]))
                clique.add(nodes[first_index])
                nodes = list(filter(lambda x: x in neighbors, nodes))
            if len(clique) > best_clique_size:
                best_clique = clique
                best_clique_size = len(clique)
        return best_clique

    @staticmethod
    def largest_first_randomized(graph: nx.Graph, n_iterations: int = 55,
                                 k_first: int = 6, **kwargs) -> set:
        best_clique = None
        best_clique_size = 0
        for _ in range(n_iterations):
            clique = set()
            nodes = [node[0] for node in sorted(nx.degree(graph), key=lambda x: x[1], reverse=True)]
            while len(nodes) > 0:
                random_index = np.random.randint(0, min(k_first, len(nodes)))
                neighbors = list(graph.neighbors(nodes[random_index]))
                clique.add(nodes[random_index])
                nodes = list(filter(lambda x: x in neighbors, nodes))
            if len(clique) > best_clique_size:
                best_clique = clique
                best_clique_size = len(clique)
        return best_clique


def test_max_clique_heuristic():
    benches = {**EASY,
               **MEDIUM,
               **HARD}
    for filepath in benches:
        print(f'Bench: {filepath}')
        G = read_graph_file(filepath, verbose=False)
        heuristic = HeuristicMaxClique(G)
        found_clique = heuristic.run()
        print(f'Found clique: {sum(found_clique)} True clique: {benches[filepath]}')


if __name__ == '__main__':
    test_max_clique_heuristic()
