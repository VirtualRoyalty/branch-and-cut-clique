import pandas as pd
from pprint import pprint

from problem import ProblemHandler
from heuristic import HeuristicMaxClique
# from branch_and_bound import BranchAndBound
from branch_and_cut import BranchAndCut
from benchmarks import EASY, MEDIUM, HARD, REST
from utils import *


def run_test(benchmark: str, abs_tol: float = 1e-4, time_limit: int = None):
    print(f'{benchmark} started...')
    graph = read_graph_file(benchmark, verbose=False)
    problem_handler = ProblemHandler(graph=graph)
    problem_handler.design_problem()
    print('Problem constructed!')
    heuristic = HeuristicMaxClique(graph)
    heuristic_clique = heuristic.run()
    heuristic_clique_size = int(sum(heuristic_clique))
    print(f'Found heuristic solution! ({heuristic_clique_size})')
    bnb_algorithm = BranchAndCut(problem=problem_handler,
                                 initial_solution=heuristic_clique, time_limit=time_limit,
                                 initial_obj_value=heuristic_clique_size, abs_tol=abs_tol)
    exec_time = bnb_algorithm.timed_run()
    _minutes, _seconds = divmod(exec_time, 60)
    clique_nodes = to_node_indexes(bnb_algorithm.best_solution)
    _result = dict(benchmark=benchmark.split('/')[-1],
                   heuristic_clique_size=heuristic_clique_size,
                   bnb_clique_size=bnb_algorithm.best_obj_value,
                   is_bnb_solution_clique=BranchAndCut.is_clique(graph, clique_nodes)[0],
                   bnb_exec_time=f'{_minutes:.0f}min {_seconds:.1f}sec',
                   bnb_exec_time_seconds=exec_time,
                   bnb_call_count=bnb_algorithm.call_counter,
                   bnb_max_recursion_depth=None,
                   )
    return _result


def run_tests(benchmarks: list, time_limit: int = None, abs_tol: float = 1e-4,
              out_folder: str = 'results/', suffix: str = ''):
    results = []
    for filepath in benchmarks:
        try:
            result_dct = run_test(filepath, abs_tol=abs_tol, time_limit=time_limit)
            result_dct['true_clique_size'] = benchmarks[filepath]
            pprint(result_dct)
            results.append(result_dct)
        except TimeoutException as timeout:
            print(filepath, timeout.msg)
            results.append(dict(benchmark=filepath.split('/')[-1],
                                bnb_exec_time=timeout.msg,
                                bnb_clique_size=timeout.best_clique_size,
                                true_clique_size=benchmarks[filepath]))
        result_df = pd.DataFrame(results)
        result_df.to_csv(out_folder + f'results_{suffix}.csv')
        result_df.to_excel(out_folder + f'results_{suffix}.xlsx')
    return


if __name__ == '__main__':
    benchmarks = dict(**EASY, **MEDIUM)
    run_tests(benchmarks=benchmarks, time_limit=3600, suffix='_improvments_cut', abs_tol=1e-4)
