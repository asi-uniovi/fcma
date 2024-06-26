"""
Performance analysis of FCMA on problems defined in JSON or PKL files.
For every JSON or PKL file it generates a row in a CSV file with the following information:
- JSON file name with the problem.
- Speed level: 1, 2 or 3.
- Cost of the solution in $/hour.
- Cost of the solution relative to the lower bound.
- Time to calculate the solution in seconds.
- Cost of the solution relative to the reference analysis.
- Time to calculate the solution relative to the reference analysis.
- Actual SFMPL metric
- Container isolation metric
- Virtual machine recycling metric
- Virtual machine load-balance metric

The reference analysis is a CSV file with "-0" suffix in the file name. On the first execution the
CSV file has "-0" suffix, which is incremented on subsequent analysis.

At the end of the sumamary a row with "total" as special name is written for every speed level.
"""

import json
import pickle
import pathlib
import csv
import datetime
from collections import defaultdict
import fcma
from fcma import Fcma
from fcma.serialization import ProblemSerializer
from fcma import PULP_CBC_CMD, FcmaStatus
from pint import set_application_registry
from cloudmodel.unified.units import ureg

set_application_registry(ureg)


file_dir = "problem_data"
perf_dir = "perf_data"
perf_file_prefix = f"{perf_dir}/perf"
perf_ref_file = f"{perf_file_prefix}-0.csv"

# A value higher than or equal to 1 that multiplies application loads
# to get virtual machine recycling metrics.
recycling_load_mul = 1.2  # Load 20 % higher


gap_rel = 0.05  # Maximum relative gap between any ILP solution and the optimal
solver = PULP_CBC_CMD(msg=0, gapRel=gap_rel)

csv_label_row = [
    "file",
    "speed_level",
    "cost($/hour)",
    "cost/lower_bound",
    "prealloc_time(sec)",
    "total_time(sec)",
    "relative_cost",
    "relative_prealloc_time",
    "relative_time",
    "fault_tolerance_m",
    "container_isolation_m",
    "vm_recycling_m",
    "vm_load_balance_m",
    "comment",
]


def csv_to_dict(csv_file_path: str) -> dict:
    """
    Read a CSV file with results.
    """
    data = {}
    with open(csv_file_path, "r") as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data[(row["file"], row["speed_level"])] = row
    return data


def get_next_perf_path() -> str:
    """
    Get the path of the new CSV file so the previous is not overwritten.
    """
    perf_files = [f.name for f in pathlib.Path(perf_dir).glob("*.csv")]
    if len(perf_files) > 0:
        perf_files.sort()
        last_number = int(perf_files[-1].split(".")[0].split("-")[-1])
        return f"{perf_file_prefix}-{last_number+1}.csv"
    return perf_ref_file


def get_perf_data(
    sol: fcma.Solution, perf_ref_data: dict, total_data: dict, lower_bound: float
) -> list:
    """
    Get performance data from the current solution and the reference performance data.
    At the same time, total performance data is updated.
    """
    perf_data = []

    total_data["n"] += 1

    perf_data.append(f"{sol.statistics.final_cost.magnitude:.4f}")
    total_data["cost($/hour)"] += sol.statistics.final_cost.magnitude

    if lower_bound is not None:
        cost_to_lower_bound = (sol.statistics.final_cost / lower_bound).magnitude
    else:
        cost_to_lower_bound = -1
    perf_data.append(f"{cost_to_lower_bound:.4f}")
    total_data["cost/lower_bound"] += cost_to_lower_bound

    perf_data.append(f"{sol.statistics.pre_allocation_seconds:.4f}")
    total_data["prealloc_time(sec)"] += sol.statistics.pre_allocation_seconds

    perf_data.append(f"{sol.statistics.total_seconds:.4f}")
    total_data["time(sec)"] += sol.statistics.total_seconds

    if len(perf_ref_data) == 0:
        perf_data.extend(["1.0", "1.0", "1.0"])
        total_data["relative_cost"] += 1.0
        total_data["relative_prealloc_time"] += 1.0
        total_data["relative_time"] += 1.0
    else:
        relative_cost = sol.statistics.final_cost.magnitude / float(perf_ref_data["cost($/hour)"])
        perf_data.append(f"{relative_cost:.4f}")
        total_data["relative_cost"] += relative_cost

        relative_prealloc_time = sol.statistics.pre_allocation_seconds / float(
            perf_ref_data["prealloc_time(sec)"]
        )
        perf_data.append(f"{relative_prealloc_time:.4f}")
        total_data["relative_prealloc_time"] += relative_prealloc_time

        relative_time = sol.statistics.total_seconds / float(perf_ref_data["total_time(sec)"])
        perf_data.append(f"{relative_time:.4f}")
        total_data["relative_time"] += relative_time

    perf_data.append(f"{sol.statistics.fault_tolerance_m:.4f}")
    total_data["fault_tolerance_m"] += sol.statistics.fault_tolerance_m

    perf_data.append(f"{sol.statistics.container_isolation_m:.4f}")
    total_data["container_isolation_m"] += sol.statistics.container_isolation_m

    perf_data.append(f"{solution.statistics.vm_recycling_m:.4f}")
    total_data["vm_recycling_m"] += sol.statistics.vm_recycling_m

    perf_data.append(f"{solution.statistics.vm_load_balance_m:.4f}")
    total_data["vm_load_balance_m"] += sol.statistics.vm_load_balance_m

    return perf_data


# Read reference analysis data from file perf_ref_file
perf_ref_data = {}
try:
    perf_ref_data = csv_to_dict(perf_ref_file)
except:
    pass

# Get the path of the CSV file with analysis to be written
perf_path = get_next_perf_path()

# Systems are in the form of JSON or PKL files
file_names = [f.name for f in pathlib.Path(file_dir).glob("*.pkl")]
file_names += [f.name for f in pathlib.Path(file_dir).glob("*.json")]

with open(perf_path, "w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(csv_label_row)
    total_data = {1: defaultdict(float), 2: defaultdict(float), 3: defaultdict(float)}
    # Perform the three FCMA analysis with each PKL or JSON file
    for file_name in file_names:
        file_path = f"{file_dir}/{file_name}"
        with open(file_path, "r") as problem_file:
            if file_path.endswith(".pkl"):
                problem = pickle.load(problem_file)
                serializer = ProblemSerializer(problem)
                fcma_problem = ProblemSerializer.from_dict(serializer.as_dict())
            else:
                fcma_problem = ProblemSerializer.from_dict(json.load(problem_file))
            if recycling_load_mul > 1.0:
                mul_workloads = {
                    key: value * recycling_load_mul for key, value in fcma_problem.workloads.items()
                }
                mul_fcma_problem = Fcma(fcma_problem.system, mul_workloads)
            # Perform analysis with the three speed levels
            for speed_level in (1, 2, 3):
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                print(f"{current_time}: {file_name}; speed level = {speed_level}")
                solution = fcma_problem.solve(
                    fcma.SolvingPars(speed_level=speed_level, solver=solver)
                )
                if recycling_load_mul > 1.0:
                    print(f"\tSolving with workload x {recycling_load_mul:.2f}")
                    mul_solution = mul_fcma_problem.solve(
                        fcma.SolvingPars(speed_level=speed_level, solver=solver)
                    )
                solution.statistics.total_seconds = solution.statistics.total_seconds
                if speed_level == 1:
                    if solution.statistics.pre_allocation_status == FcmaStatus.INVALID:
                        lower_bound = None
                    else:
                        lower_bound = solution.statistics.pre_allocation_lower_bound_cost
                if recycling_load_mul > 1.0:
                    solution.statistics.update_metrics(solution.allocation, mul_solution.allocation)
                else:
                    solution.statistics.update_metrics(solution.allocation)
                reference_data = {}
                if (file_name, str(speed_level)) in perf_ref_data:
                    reference_data = perf_ref_data[(file_name, str(speed_level))]
                perf_data = get_perf_data(
                    solution, reference_data, total_data[speed_level], lower_bound
                )
                perf_data.insert(0, speed_level)
                perf_data.insert(0, file_name)
                perf_data.append("")  # For comments
                csv_writer.writerow(perf_data)
    if len(file_names) > 0:
        # Write total rows in the CSV file
        for speed_level in (1, 2, 3):
            total_data_row = [
                "total",
                speed_level,
                f'{total_data[speed_level]["cost($/hour)"]:.4f}',
                f'{total_data[speed_level]["cost/lower_bound"] / total_data[speed_level]["n"]:.4f}',
                f'{total_data[speed_level]["prealloc_time(sec)"]:.4f}',
                f'{total_data[speed_level]["time(sec)"]:.4f}',
                f'{total_data[speed_level]["relative_cost"] / total_data[speed_level]["n"]:.4f}',
                f'{total_data[speed_level]["relative_prealloc_time"] / total_data[speed_level]["n"]:.4f}',
                f'{total_data[speed_level]["relative_time"] / total_data[speed_level]["n"]:.4f}',
                f'{total_data[speed_level]["fault_tolerance_m"] / total_data[speed_level]["n"]:.4f}',
                f'{total_data[speed_level]["container_isolation_m"] / total_data[speed_level]["n"]:.4f}',
                f'{total_data[speed_level]["vm_recycling_m"] / total_data[speed_level]["n"]:.4f}',
                f'{total_data[speed_level]["vm_load_balance_m"] / total_data[speed_level]["n"]:.4f}',
                "",
            ]
            csv_writer.writerow(total_data_row)
