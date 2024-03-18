import logging
from cloudmodel.unified.units import ComputationalUnits, RequestsPerTime, Storage
import aws_eu_west_1
from fcma import App, AppFamilyPerf, System, Fcma, SolvingPars
from fcma.visualization import SolutionPrinter

# Set logging level
logging.basicConfig(level=logging.INFO)

apps = {"appA": App(name="appA")}
workloads = {apps["appA"]: RequestsPerTime("20  req/s")}
system: System = {
    (apps["appA"], aws_eu_west_1.c5_m5_r5_fm): AppFamilyPerf(
        cores=ComputationalUnits("400000 mcores"),
        mem=Storage("500 mebibytes"),
        perf=RequestsPerTime("0.4 req/s"),
    ),
}
problem = Fcma(system, workloads)
solution = problem.solve()

# Print results
SolutionPrinter(solution).print()

# Check the solution
slack = problem.check_allocation()
print("\n----------- Solution check --------------")
for attribute in dir(slack):
    if attribute.endswith("percentage"):
        print(f"{attribute}: {getattr(slack, attribute): .2f} %")
print("-----------------------------------------")
