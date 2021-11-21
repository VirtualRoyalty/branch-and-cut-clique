import time
import numpy as np
import networkx as nx
from math import isclose

import utils
from problem import ProblemHandler
from separator import greedy_separation, IteratedLocalSearch


class BranchAndCut:

    def __init__(self, problem: ProblemHandler, initial_obj_value: float, initial_solution: list,
                 abs_tol: float = 1e-4, time_limit: int = None):
        self.call_counter = 0
        self.problem = problem
        self.best_obj_value = initial_obj_value
        self.best_solution = initial_solution
        self.num_of_nodes = problem.graph.number_of_nodes()
        self.constrained_vars = np.zeros(self.num_of_nodes, dtype=np.bool)
        self.constraint_size = 0
        self.abs_tol = abs_tol
        self.start_time = None
        self.time_limit = time_limit
        self.max_sep_iter = 1000

    @utils.timer
    def timed_run(self):
        if self.time_limit:
            self.start_time = time.perf_counter()
        self.run()
        return

    def run(self):
        # timeout stop
        if self.time_limit:
            elapsed_time = time.perf_counter() - self.start_time
            if elapsed_time > self.time_limit:
                raise utils.TimeoutException(best_clique_size=self.best_obj_value,
                                             msg=f'TIMEOUT: >{round(elapsed_time)}s elapsed')
        self.call_counter += 1
        current_obj_value = self.problem.solve_problem()
        if current_obj_value is None:
            return
        if int(current_obj_value + self.abs_tol) <= self.best_obj_value:
            return
        current_solution = self.problem.get_solution()
        if self.call_counter % 100 == 0:
            elapsed_time = time.perf_counter() - self.start_time
            _minutes, _seconds = divmod(elapsed_time, 60)
            print(f'{_minutes:.0f}min {_seconds:.1f}sec ({round(elapsed_time, 1)})')
            self.problem.distill_constraints()

        not_improve_criteria = 0
        obj_value_history = list()
        for sep_iter in range(self.max_sep_iter):
            violated_constraints = IteratedLocalSearch.get_ind_set(self.problem.graph, current_solution, first_k=15)
            # violated_constraints = greedy_separation(self.problem.graph, current_solution, first_k=20)
            if len(violated_constraints) == 0:
                break
            ils = IteratedLocalSearch(self.problem.graph, violated_constraints[0][0], weights=current_solution)
            ils.run(n_iter=100, verbose=False)
            violated_constraints.append((ils.best_sol_set, ils.best_set_weight))
            added_flag = False
            for itr, max_weigh_constraint in enumerate(violated_constraints):
                constraint, weight = max_weigh_constraint
                if weight > (1.0 + self.abs_tol):
                    added_flag = True
                    self.problem.add_constraint(var_names=[f'x{i}' for i in constraint], sense='L',
                                                constraint_name=f'Strong{sep_iter}_{itr}')
            if not added_flag:
                break
            current_obj_value = self.problem.solve_problem()
            if current_obj_value is None:
                return
            if int(current_obj_value + self.abs_tol) <= self.best_obj_value:
                return
            if len(obj_value_history) > 0:
                if isclose(obj_value_history[-1], current_obj_value, abs_tol=1e-2):
                    break
                if (obj_value_history[-1] - current_obj_value) < 0.1:
                    not_improve_criteria += 1
                else:
                    not_improve_criteria = 0
                if not_improve_criteria > 10:
                    break
            obj_value_history.append(current_obj_value)
            current_solution = self.problem.get_solution()

        branching_var_index = self.select_branching_var(current_solution)
        if branching_var_index == -1:
            weak_constraints = self.check_solution(current_solution)
            if weak_constraints != -1:
                for itr, pair in enumerate(weak_constraints):
                    var_names = [f'x{i}' for i in pair]
                    self.problem.add_constraint(var_names=var_names, sense='L',
                                                constraint_name=f'Weak{self.call_counter}_{itr}')
                self.run()
            else:
                print(f'\t\t\tFound new best: {current_obj_value}')
                self.best_solution = current_solution
                self.best_obj_value = round(current_obj_value)
                return
        else:
            branching_var_name = f'x{branching_var_index}'
            rounded_value = round(current_solution[branching_var_index])
            for branch_value in [rounded_value, 1 - round(rounded_value)]:
                constraint_name = f'Branch{self.call_counter}_{branch_value}_{branching_var_name}'
                self.problem.add_constraint(var_names=[branching_var_name], sense='E',
                                            constraint_name=constraint_name, rhs=branch_value)
                self.constrained_vars[branching_var_index] = True
                self.constraint_size += 1
                self.run()
                self.problem.remove_constraint(constraint_name)
                self.constrained_vars[branching_var_index] = False
                self.constraint_size -= 1
        return

    def check_solution(self, solution: list) -> list:
        clique_nodes = utils.to_node_indexes(solution, abs_tol=self.abs_tol)
        clique_flag, subgraph = self.is_clique(self.problem.graph, clique_nodes)
        if clique_flag:
            return -1
        else:
            return self.problem.get_complement_edges(subgraph)

    @staticmethod
    def is_clique(graph: nx.Graph, nodes: list) -> bool:
        subgraph: nx.Graph = graph.subgraph(nodes)
        num_of_nodes = subgraph.number_of_nodes()
        num_of_edges = subgraph.number_of_edges()
        num_of_edges_complete = int(num_of_nodes * (num_of_nodes - 1) / 2)
        if num_of_edges == num_of_edges_complete:
            return True, None
        return False, subgraph

    def select_branching_var(self, solution: list) -> int:
        selected_var_index = -1
        min_diff_to_one = 2
        for _index, value in enumerate(solution):
            if self.constrained_vars[_index] == 0:
                if not self.is_integer(value, abs_tol=self.abs_tol):
                    diff_to_one = abs(1 - value)
                    if diff_to_one < min_diff_to_one:
                        min_diff_to_one = diff_to_one
                        selected_var_index = _index
        return selected_var_index

    @staticmethod
    def is_all_integer(variables: list, abs_tol: float = 1e-4) -> bool:
        for var in variables:
            if not BranchAndCut.is_integer(var, abs_tol=abs_tol):
                return False
        return True

    @staticmethod
    def is_integer(var: float, abs_tol: float = 1e-4) -> bool:
        return isclose(var, 0, abs_tol=abs_tol) or isclose(var, 1, abs_tol=abs_tol)
