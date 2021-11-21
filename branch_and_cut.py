import time
import numpy as np
import networkx as nx
from math import isclose

import utils
from problem import ProblemHandler
from separator import simple_separation, IteratedLocalSearch


class BranchAndCut:

    def __init__(self, problem: ProblemHandler, initial_obj_value: float, initial_solution: list,
                 abs_tol: float = 1e-4, time_limit: int = None):
        self.call_counter = 0
        self.recursion_depth = 0
        self.max_recursion_depth = 0
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
        if self.time_limit:
            elapsed_time = time.perf_counter() - self.start_time
            if elapsed_time > self.time_limit:
                raise utils.TimeoutException(best_clique_size=self.best_obj_value,
                                             msg=f'TIMEOUT: >{round(elapsed_time)}s elapsed')
        self.call_counter += 1
        current_obj_value = self.problem.solve_problem()
        # print('CURRENT_OBJ_VALUE', current_obj_value)
        if current_obj_value is None:
            return
        current_solution = self.problem.get_solution()

        if self.call_counter % 40 == 0:
            print('CONSTR NUM:', self.problem.problem.linear_constraints.get_num())
            self.problem.distill_constraints()
            print('*after distill* CONSTR NUM:', self.problem.problem.linear_constraints.get_num())

        if int(current_obj_value + self.abs_tol) <= self.best_obj_value:
            return
        obj_value_history = list()
        not_improve_criteria = 0
        for sep_iter in range(self.max_sep_iter):
            obj_value_history.append(current_obj_value)
            violated_constraints = simple_separation(self.problem.graph, current_solution,
                                                     n_iter=25, first_k=7)
            if len(violated_constraints) == 0:
                # obj_value_history.append('NO_CONSTRAINTS')
                break
            start_solution = np.zeros(self.num_of_nodes, dtype=bool)
            for node in violated_constraints[0][0]:
                start_solution[node] = 1
            # ils = IteratedLocalSearch(self.problem.graph, start_solution, weights=current_solution)
            # for _ in range(1):
            #     ils.run(n_iter=300, verbose=False)
            #     violated_constraints.append((ils.best_sol_set, ils.best_set_weight))
            #     ils.reset()
            for itr, max_weigh_constraint in enumerate(violated_constraints):
                constraint, weight = max_weigh_constraint
                self.problem.add_constraint(var_names=[f'x{i}' for i in constraint], sense='L',
                                            constraint_name=f'Strong{sep_iter}_{itr}')
            previous_obj_value = current_obj_value
            current_obj_value = self.problem.solve_problem()
            if current_obj_value is None:
                return
            if int(current_obj_value + self.abs_tol) <= self.best_obj_value:
                return
            if isclose(previous_obj_value, current_obj_value, abs_tol=1e-2):
                break
            if (previous_obj_value - current_obj_value) < 0.5:
                not_improve_criteria += 1
            else:
                not_improve_criteria = 0
            if not_improve_criteria > 3:
                # obj_value_history.append('NOT_IMPROVE')
                break
            current_solution = self.problem.get_solution()
            # if len(obj_value_history) > 5:
            #     if np.mean(obj_value_history[-5:-1]) < obj_value_history[-1] + 0.01:
            #         obj_value_history.append('NOT_IMPROVE')
            #         break

        if self.call_counter % 50 == 0 or self.call_counter <= 1:
            print('OBJ_HISTORY:', np.around(obj_value_history, 2))
            print('CURRENT SOLUTION', np.around(current_solution, 2))

        branching_var_index = self.select_branching_var(current_solution)
        if branching_var_index == -1:
            # print('~~~~N0 BRANCHING VAR~~~~')
            weak_constraints = self.check_solution(current_solution)
            if weak_constraints != -1:
                print('----' * 5 + 'WEAK CONSTRAINTS:', len(weak_constraints))
                print('~~~~~~~~~~~~~~~~~', weak_constraints[:5])
                # print('WEAK CONSTRAINTS:', weak_constraints)
                for itr, pair in enumerate(weak_constraints):
                    var_names = [f'x{i}' for i in pair]
                    # print(var_names)
                    # print('VIOLATED VARS:', current_solution[pair[0]-1], current_solution[pair[1]-1])
                    self.problem.add_constraint(var_names=var_names, sense='L',
                                                constraint_name=f'Weak{self.call_counter}_{itr}')
                # print('CONSTR NUM:', self.problem.problem.linear_constraints.get_num())
                # print(self.problem.problem.linear_constraints.get_rows()[-5:])
                # print(self.problem.problem.linear_constraints.get_rhs()[-5:])
                # print(self.problem.problem.linear_constraints.get_histogram())
                self.run()
            else:
                print(f'******************Found new best: {current_obj_value}**************************')
                self.best_solution = current_solution
                self.best_obj_value = round(current_obj_value)
                return
        else:
            # branching_var_name = f'x{branching_var_index + 1}'
            branching_var_name = f'x{branching_var_index}'
            rounded_value = round(current_solution[branching_var_index])
            # print(f'Branching {branching_var_name} to {rounded_value} ... call{self.call_counter}')
            for branch_value in [1, 0]:  # [rounded_value, 1 - round(rounded_value)]:
                constraint_name = f'Branch{self.call_counter}_{branch_value}_{branching_var_name}'
                self.problem.add_constraint(var_names=[branching_var_name], sense='E',
                                            constraint_name=constraint_name, rhs=branch_value)
                self.constrained_vars[branching_var_index] = 1
                self.constraint_size += 1
                self.recursion_depth += 1
                self.max_recursion_depth = max(self.recursion_depth, self.max_recursion_depth)
                self.run()
                self.problem.remove_constraint(constraint_name)
                self.constrained_vars[branching_var_index] = 0
                self.constraint_size -= 1
                self.recursion_depth -= 1
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
