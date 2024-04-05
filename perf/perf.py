"""
Performance analysis of FCMA on problems defined in JSON files.
For every JSON file it generates a row in a CSV file with the following information:
- JSON file name with the problem.
- Speed level: 1, 2 or 3.
- Cost of the solution in $/hour.
- Cost of the solution relative to the lower bound.
- Time to calculate the solution in seconds.
- Cost of the solution relative to the reference analysis.
- Time to calculate the solution relative to the reference analysis.

The reference analysis is a CSV file with "-0" suffix in the file name. On the first execution the
CSV file has "-0" suffix, which is incremented on subsequent analysis.

At the end of the sumamary a row with "total" as special name is written for every speed level.
"""

import json
import pathlib
import csv
import datetime
import fcma
from fcma.serialization import ProblemSerializer
from fcma import PULP_CBC_CMD, FcmaStatus

json_dir = "json_data"
perf_dir = "perf_data"
perf_file_prefix = f"{perf_dir}/perf"
perf_ref_file = f"{perf_file_prefix}-0.csv"

gap_rel = 0.02  # Maximum relative gap between enay ILP solution and the optimal
solver = PULP_CBC_CMD(msg=0, gapRel=gap_rel)

csv_label_row = [
    "file",
    "speed_level",
    "cost($/hour)",
    "cost/lower_bound",
    "time(sec)",
    "relative_cost",
    "relative_time",
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
    Get performance data from the current solution nd te reference performance data.
    At the same time, total performance data is updated.
    """
    perf_data = []

    if "n" not in total_data:
        total_data["n"] = 1
    else:
        total_data["n"] += 1

    perf_data.append(f"{sol.statistics.final_cost.magnitude:.4f}")
    if "cost($/hour)" not in total_data:
        total_data["cost($/hour)"] = sol.statistics.final_cost.magnitude
    else:
        total_data["cost($/hour)"] += sol.statistics.final_cost.magnitude

    if lower_bound is not None:
        cost_to_lower_bound = sol.statistics.final_cost.magnitude / lower_bound
    else:
        cost_to_lower_bound = -1
    perf_data.append(f"{cost_to_lower_bound:.4f}")
    if "cost/lower_bound" not in total_data:
        total_data["cost/lower_bound"] = cost_to_lower_bound
    else:
        total_data["cost/lower_bound"] += cost_to_lower_bound

    perf_data.append(f"{sol.statistics.total_seconds:.4f}")
    if "time(sec)" not in total_data:
        total_data["time(sec)"] = sol.statistics.total_seconds
    else:
        total_data["time(sec)"] += sol.statistics.total_seconds

    if len(perf_ref_data) == 0:
        perf_data.extend(["1.0", "1.0"])
        if "relative_cost" not in total_data:
            total_data["relative_cost"] = 1.0
        else:
            total_data["relative_cost"] += 1.0
        if "relative_time" not in total_data:
            total_data["relative_time"] = 1.0
        else:
            total_data["relative_time"] += 1.0

    else:
        relative_cost = sol.statistics.final_cost.magnitude / float(perf_ref_data["cost($/hour)"])
        perf_data.append(f"{relative_cost:.4f}")
        if "relative_cost" not in total_data:
            total_data["relative_cost"] = relative_cost
        else:
            total_data["relative_cost"] += relative_cost

        relative_time = sol.statistics.total_seconds / float(perf_ref_data["time(sec)"])
        perf_data.append(f"{relative_time:.4f}")
        if "relative_time" not in total_data:
            total_data["relative_time"] = relative_time
        else:
            total_data["relative_time"] += relative_time

    return perf_data


# Read reference analysis data from file perf_ref_file
perf_ref_data = {}
try:
    perf_ref_data = csv_to_dict(perf_ref_file)
except:
    pass

# Get the path of the CSV file with analysis to be written
perf_path = get_next_perf_path()

# Systems are in the form of JSON files
file_names = [f.name for f in pathlib.Path(json_dir).glob("*.json")]

with open(perf_path, "w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(csv_label_row)
    total_data = {1: {}, 2: {}, 3: {}}
    # Perform the three FCMA analysis with each JSON file
    for json_file_name in file_names:
        json_path = f"{json_dir}/{json_file_name}"
        with open(json_path, "r") as json_file:
            # Perform analysis with the three speed levels
            fcma_problem = ProblemSerializer.from_dict(json.load(json_file))
            for speed_level in (1, 2, 3):
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                print(f"{current_time}: {json_file_name}; speed level = {speed_level}")
                solution = fcma_problem.solve(
                    fcma.SolvingPars(speed_level=speed_level, solver=solver)
                )
                if speed_level == 1:
                    lower_bound = solution.statistics.pre_allocation_cost.magnitude
                    if solution.statistics.pre_allocation_status == FcmaStatus.FEASIBLE:
                        lower_bound *= 1.0 + gap_rel
                    elif solution.statistics.pre_allocation_status == FcmaStatus.INVALID:
                        lower_bound = None
                reference_data = {}
                if (json_file_name, str(speed_level)) in perf_ref_data:
                    reference_data = perf_ref_data[(json_file_name, str(speed_level))]
                perf_data = get_perf_data(
                    solution, reference_data, total_data[speed_level], lower_bound
                )
                perf_data.insert(0, speed_level)
                perf_data.insert(0, json_file_name)
                csv_writer.writerow(perf_data)
    if len(file_names) > 0:
        # Write total rows in the CSV file
        for speed_level in (1, 2, 3):
            total_data_row = [
                "total",
                speed_level,
                f'{total_data[speed_level]["cost($/hour)"]:.4f}',
                f'{total_data[speed_level]["cost/lower_bound"] / total_data[speed_level]["n"]:.4f}',
                f'{total_data[speed_level]["time(sec)"]:.4f}',
                f'{total_data[speed_level]["relative_cost"] / total_data[speed_level]["n"]:.4f}',
                f'{total_data[speed_level]["relative_time"] / total_data[speed_level]["n"]:.4f}',
            ]
            csv_writer.writerow(total_data_row)
