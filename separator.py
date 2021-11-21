import math
import random
import numpy as np
import networkx as nx
from tqdm import tqdm

import utils


def simple_separation(graph: nx.Graph, solution: list,
                      first_k: int = 10, abs_tol=1e-3, **kwargs) -> list:
    coloring_dct = nx.coloring.greedy_color(graph, strategy=nx.coloring.strategy_random_sequential)
    independent_sets = list()
    for index in range(1, first_k):
        random_index = np.random.randint(0, len(solution))
        ind_set = set()
        set_weight = 0
        for _node, color in coloring_dct.items():
            if color == coloring_dct[random_index]:
                ind_set.add(_node)
                set_weight += solution[_node]
        if len(ind_set) > 2:  # and set_weight > (1 + abs_tol):
            independent_sets.append((ind_set, set_weight))
    independent_sets = sorted(independent_sets, key=lambda item: (item[1], len(item[0])), reverse=True)
    return independent_sets[:first_k]  # random.sample(independent_sets, k=min(len(independent_sets), first_k))


def greedy_separation(graph: nx.Graph, solution: list,
                      n_iter: int = 20, first_k: int = 10, abs_tol=1e-3) -> list:
    solution_indexes = np.argsort(solution)
    independent_sets = list()
    for _ in range(n_iter):
        coloring_dct = nx.coloring.greedy_color(graph, strategy=nx.coloring.strategy_random_sequential)
        for index in range(1, min(first_k, len(solution_indexes) - 1)):
            max_w_index = solution_indexes[-index]
            ind_set = set()
            set_weight = 0
            for _node, color in coloring_dct.items():
                if color == coloring_dct[max_w_index]:
                    ind_set.add(_node)
                    set_weight += solution[_node]
            if len(ind_set) > 2 and set_weight > (1 + abs_tol):
                independent_sets.append((ind_set, set_weight))
    independent_sets = sorted(independent_sets, key=lambda item: (item[1], len(item[0])), reverse=True)
    return independent_sets[:first_k]


class LocalSearch:
    def __init__(self, graph: nx.Graph, weights: list):
        self.graph = graph
        self.num_of_nodes = self.graph.number_of_nodes()
        self.weights = weights
        return

    def random_insert(self, sol_set, others_set):
        candidates = random.sample(others_set, k=min(5, len(others_set)))
        c_index = np.argmax(list(map(lambda x: self.weights[x], candidates)))
        node_to_add = candidates[c_index]
        new_sol_set = set()
        new_others_set = set()
        new_set_weight = self.weights[node_to_add]
        new_sol_set.add(node_to_add)
        for node_in_set in sol_set:
            if self.graph.has_edge(node_to_add, node_in_set):
                new_others_set.add(node_in_set)
            else:
                if node_in_set not in new_sol_set:
                    new_sol_set.add(node_in_set)
                    new_set_weight += self.weights[node_in_set]
        new_others_set = new_others_set.union(others_set - set([node_to_add]))
        return new_sol_set, new_others_set, new_set_weight

    def add_free_nodes(self, sol_set: set, others_set: set, set_weight: float, sample_size=200):
        candidates = random.sample(others_set, k=min(sample_size, len(others_set)))
        for candidate in set(candidates):
            flag = True
            for node_in_set in sol_set:
                if self.graph.has_edge(candidate, node_in_set):
                    flag = False
                    break
            if flag:
                if candidate not in sol_set:
                    sol_set.add(candidate)
                    set_weight += self.weights[candidate]
        return set_weight

    def run(self, sol_set: set, others_set: set, set_weight: float, stop_k: int = 5):
        k = 1
        current_sol_set = sol_set.copy()
        current_others_set = others_set.copy()
        while k <= stop_k:
            new_sol, new_others, new_weight = self.random_insert(current_sol_set,
                                                                 current_others_set)
            if new_weight <= set_weight:
                k += 1
            else:
                k = 1
                set_weight = new_weight
                current_sol_set = new_sol
                current_others_set = new_others
                set_weight = self.add_free_nodes(current_sol_set,
                                                 current_others_set,
                                                 set_weight)
        return current_sol_set, current_others_set, set_weight


class IteratedLocalSearch:
    def __init__(self, graph: nx.Graph, solution: np.ndarray, weights: list):
        self.weights = weights
        self.graph = graph
        self.num_of_nodes = self.graph.number_of_nodes()
        sol_set, others_set, weight = self.init_solution_sets(solution)
        self.best_sol_set = sol_set
        self.best_others_set = others_set
        self.best_set_weight = weight
        self.init_sol_set = sol_set.copy()
        self.init_others_set = others_set.copy()
        self.init_set_weight = weight
        self.searcher = LocalSearch(graph, weights)
        return

    def reset(self):
        self.best_sol_set = self.init_sol_set
        self.best_others_set = self.init_others_set
        self.best_set_weight = self.init_set_weight

    def init_solution_sets(self, initial_solution: set):
        solution_weight = 0
        sol = initial_solution.copy()
        others = set()
        for node in self.graph.nodes():
            if node not in sol:
                others.add(node)
            else:
                solution_weight += self.weights[node]
        return sol, others, solution_weight

    @utils.timer
    def timed_run(self, *args, **kwargs):
        self.run(*args, **kwargs)
        return

    def run(self, n_iter: int = 50, verbose=False):
        local_sol, local_others, local_weight = self.searcher.run(self.best_sol_set,
                                                                  self.best_others_set,
                                                                  self.best_set_weight)
        improvement = 5
        for itr in tqdm(range(n_iter), disable=not verbose):
            sol, others, weight = self.searcher.random_insert(local_sol, local_others)
            sol, others, weight = self.searcher.run(sol, others, weight)

            if self.best_set_weight < weight:
                self.best_sol_set = sol
                self.best_others_set = others
                self.best_set_weight = weight
                local_sol = sol
                local_others = others
            else:
                improvement -= 1
            if improvement < 0:
                local_sol = sol
                local_others = others
        if not math.isclose(self.best_set_weight,
                            sum(self.weights[s] for s in self.best_sol_set), abs_tol=1e3):
            raise "WEIHTS NOT EQUAL"
        sub = self.graph.subgraph(self.best_sol_set)
        if sub.number_of_edges() > 0:
            raise "NOT INDEP SET"
        return


class FastLocalSearch:
    def __init__(self, graph: nx.Graph, solution: np.array, weights: list):
        self.graph = graph
        self.num_of_nodes = self.graph.number_of_nodes()
        self.solution = solution
        self.weights = weights
        self.tightness = np.array(weights, dtype=np.float)
        self.qco = np.zeros(self.num_of_nodes, dtype=np.int)
        self.index_qco = np.array(range(0, self.num_of_nodes), dtype=np.int)
        self.q = 0
        self.c = self.num_of_nodes
        self.init_to_qco()
        return

    def swap_qco(self, i, l):
        j = self.qco[l]
        self.qco[self.index_qco[i]], self.qco[self.qco[l]] \
            = self.qco[self.qco[l]], self.qco[self.index_qco[i]]
        self.index_qco[i], self.index_qco[j] = self.index_qco[j], self.index_qco[i]
        return

    def init_qco(self):
        for node_index, in_solution in enumerate(self.solution):
            if not in_solution:
                continue
        self.insert_to_qco(node_index)
        return

    def insert_to_qco(self, node_index):
        self.swap_qco(node_index, self.q)
        self.q += 1
        self.c -= 1
        for neigh in self.graph.neighbors(node_index + 1):
            if self.tightness[neigh - 1] > 0:
                self.swap_qco(neigh - 1, self.c)
                self.c -= 1
                self.tightness[neigh - 1] -= self.weights[node_index]
        return

    def run(self, n_iter=10):
        for i in n_iter:
            qco_i = np.random.randint(self.q + 1, self.q + self.c + 1)
            node_i = self.qco[qco_i]
            if self.solution[node_i] == 0 and self.mu[node_i] > 0:
                self.solution[node_i] == 0
                for neigh in self.graph.neighbors(node_i + 1):
                    if self.solution[neigh - 1] == 1:
                        self.solution[neigh - 1] = 0
                        self.mu[neigh - 1] -= self.weights[node_i]
        return


if __name__ == '__main__':
    from utils import read_graph_file

    G = read_graph_file('benchmarks/DIMACS_all_ascii/keller4.clq', verbose=False)

    from problem import ProblemHandler

    ph = ProblemHandler(G)
    ph.design_problem(accepted_sets_ratio=0.15)
    ph.solve_problem()
    solution = ph.get_solution()
    ind_sets = simple_separation(ph.graph, solution, n_iter=1, first_k=20)
    print('Simple separator:', *ind_sets[:10], sep='\n')
    sub = G.subgraph(ind_sets[0][0])
    print('\tCheck:', True if sub.number_of_edges() == 0 else False, '\n')

    ils = IteratedLocalSearch(G, ind_sets[0][0], weights=solution)
    for i in range(5):
        exec_time = ils.timed_run(n_iter=200, verbose=False)
        _minutes, _seconds = divmod(exec_time, 60)
        print('Solution weights:', sum([solution[x] for x in ils.best_sol_set]))
        print('RESULT', ils.best_sol_set, 'SET WEIGHT', ils.best_set_weight)
        print(f'ILS exec time: {_minutes:.0f}min {_seconds:.1f}sec')
        sub = G.subgraph(ils.best_sol_set)
        print('\tCheck:', True if sub.number_of_edges() == 0 else False)
        ils.reset()
