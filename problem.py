import cplex
import networkx as nx

from utils import NotDefinedException


class ProblemHandler:
    problem: cplex.Cplex
    STRATEGIES = [
        nx.coloring.strategy_largest_first,
        nx.coloring.strategy_random_sequential,
        nx.coloring.strategy_independent_set,
        nx.coloring.strategy_connected_sequential_bfs,
        nx.coloring.strategy_saturation_largest_first,
    ]

    def __init__(self, graph: nx.Graph, is_integer: bool = False, verbose: bool = False):
        self.problem = None
        self.graph = graph
        self.is_integer = is_integer
        self.verbose = verbose
        return

    def solve_problem(self) -> float:
        if self.problem:
            try:
                self.problem.solve()
                return self.problem.solution.get_objective_value()
            except cplex.exceptions.CplexSolverError as error:
                print(error)
                return None
        else:
            raise NotDefinedException

    def get_solution(self) -> list:
        if self.problem:
            return self.problem.solution.get_values()
        else:
            raise NotDefinedException

    def get_slack_values(self) -> list:
        if self.problem:
            return self.problem.solution.get_linear_slacks()
        else:
            raise NotDefinedException

    def add_constraint(self, var_names: str, constraint_name: str,
                       rhs: float = 1.0, sense='E'):
        constraint = [var_names, [1.0] * len(var_names)]
        if self.problem:
            self.problem.linear_constraints.add(lin_expr=[constraint], senses=[sense],
                                                rhs=[rhs], names=[constraint_name])
            return
        else:
            raise NotDefinedException

    def remove_constraint(self, constraint_name):
        if self.problem:
            self.problem.linear_constraints.delete(constraint_name)
            return
        else:
            raise NotDefinedException

    def distill_constraints(self, threshold: float = 1e-3):
        if self.problem:
            slacks = self.problem.solution.get_linear_slacks()
            constraint_names = self.problem.linear_constraints.get_names()
            for i, slack in enumerate(slacks):
                if slack > threshold:
                    name = constraint_names[i]
                    if 'Branch' not in name:
                        self.remove_constraint(name)
            return
        else:
            raise NotDefinedException

    def design_problem(self, accepted_sets_ratio: float = 0.4, n_iter: int = 150):
        # specify numeric type for ILP/LP problem
        one = 1 if self.is_integer else 1.0
        zero = 0 if self.is_integer else 0.0
        independent_sets = self.get_independent_sets(self.graph, accepted_ratio=accepted_sets_ratio,
                                                     n_iter=n_iter, strategies=self.STRATEGIES)
        print(f'IND SETS: {len(independent_sets)}')
        nodes = sorted(self.graph.nodes())
        n_vars = self.graph.number_of_nodes()
        n_constraints = len(independent_sets)
        upper_bounds = [one] * n_vars
        lower_bounds = [zero] * n_vars
        obj = [one] * n_vars
        var_names = [f'x{i}' for i in nodes]
        constraint_names = [f'c{i + 1}' for i in range(n_constraints)]
        constraint_senses = ['L'] * n_constraints
        right_hand_side = [one] * n_constraints
        problem = cplex.Cplex()
        problem.variables.add(obj=obj, names=var_names, ub=upper_bounds, lb=lower_bounds)
        constraints = []
        for ind_set in independent_sets:
            constraints.append([[f'x{i}' for i in ind_set], [1.0] * len(ind_set)])
        _type = problem.variables.type.binary if self.is_integer else problem.variables.type.continuous
        for node in nodes:
            problem.variables.set_types(f'x{node}', _type)
        problem.linear_constraints.add(lin_expr=constraints, senses=constraint_senses,
                                       rhs=right_hand_side, names=constraint_names)
        problem.objective.set_sense(problem.objective.sense.maximize)
        self.problem = problem
        self.set_verbosity()
        return

    def set_verbosity(self):
        if not self.verbose:
            self.problem.set_log_stream(None)
            self.problem.set_results_stream(None)
            self.problem.set_warning_stream(None)
            self.problem.set_error_stream(None)

    @staticmethod
    def get_complement_edges(graph: nx.Graph) -> list:
        complement_g = nx.complement(graph)
        return list(filter(lambda pair: pair[0] != pair[1], complement_g.edges()))

    @staticmethod
    def get_independent_sets(graph: nx.Graph, strategies: list, n_iter: int = 35,
                             accepted_ratio: float = 0.35, min_set_size: int = 3,
                             **kwargs) -> list:
        independent_sets = set()
        for strategy in strategies:
            if strategy == nx.coloring.strategy_random_sequential:
                _n_iter = n_iter
            else:
                _n_iter = 1
            for _ in range(_n_iter):
                coloring_dct = nx.coloring.greedy_color(graph, strategy=strategy)
                color2nodes = dict()
                for node, color in coloring_dct.items():
                    if color not in color2nodes:
                        color2nodes[color] = []
                    color2nodes[color].append(node)
                for color, colored_nodes in color2nodes.items():
                    if len(colored_nodes) >= min_set_size:
                        colored_nodes = tuple(sorted(colored_nodes))
                        independent_sets.add(colored_nodes)
        # store in each ind_set in set() for faster pair constraint filtering
        independent_sets = [set(ind_set) for ind_set in independent_sets]
        independent_sets = sorted(independent_sets, key=lambda _set: len(_set), reverse=True)
        first_k = int(len(independent_sets) * accepted_ratio)
        return independent_sets[:first_k]
