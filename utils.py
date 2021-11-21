import time
import math
import argparse
import functools
import networkx as nx


def to_node_indexes(solution: list, abs_tol: float = 1e-5) -> list:
    # transform [1, 0,  1,... 1] to [0, 2, ... 10]
    return [var_index for var_index, var in enumerate(solution)
            if math.isclose(var, 1, abs_tol=abs_tol)]


def read_graph_file(file_path: str, verbose: bool = True) -> nx.Graph:
    edges = []
    file = open(file_path, 'r')
    for line in file:
        if line.startswith('c'):  # graph description
            if verbose:
                print(*line.split()[1:])
        elif line.startswith('p'):
            _, _, n_nodes, n_edges = line.split()
            if verbose:
                print(f'Nodes:  {n_nodes} Edges: {n_edges}')
        elif line.startswith('e'):
            _, node_i, node_j = line.split()
            edges.append((int(node_i) - 1, int(node_j) - 1))
        else:
            continue
    g = nx.Graph(edges)
    assert int(n_nodes) == g.number_of_nodes()
    assert int(n_edges) == g.number_of_edges()
    return g


class TimeoutException(Exception):
    def __init__(self, best_clique_size: int, msg: str = 'TIME OUT!'):
        self.msg = msg
        self.best_clique_size = best_clique_size


class NotDefinedException(Exception):
    def __init__(self, msg: str = 'Problem is not constructed yet'):
        self.msg = msg


def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        _ = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        return elapsed_time

    return wrapper_timer


def main_arg_parser():
    parser = argparse.ArgumentParser(description='Solve maximum clique problem by CPLEX solver')
    parser.add_argument('--filepath', type=str, required=True,
                        help='Path to DIMACS-format file')
    parser.add_argument('--abs_tol', type=float, required=False,
                        default=1e-5, help='absolute tolerance value for comparative operations')
    parser.add_argument('--verbose', type=bool, default=False,
                        help='Whether to stream solver logs')
    return parser.parse_args()
