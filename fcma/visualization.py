"""This module provides ways of visualizing problems and solutions for FCMA."""

from rich.console import Console
from rich.table import Table, Column
from rich import print  # pylint: disable=redefined-builtin
from cloudmodel.unified.units import Requests
from .model import IcAllocationSolution, FcmaStatus, Problem


class SolutionPrettyPrinter:
    """Utility methods to create pretty presentations of solutions."""

    def __init__(self, sol: IcAllocationSolution):
        self.sol = sol
        self.console = Console()

    def get_summary(self) -> str:
        """Returns a summary of the solution."""
        if self.is_infeasible_sol():
            return f"Non feasible solution. [bold red]{self.sol.solving_stats.ilp_status}"

        res = f"\nTotal cost: {self.sol.cost:.6f}"

        return res

    def print(self):
        """Prints tables and a summary of the solution."""
        if self.is_infeasible_sol():
            print(f"Non feasible solution. [bold red]{self.sol.solving_stats.ilp_status}")
            return

        print(self.get_ic_table())
        print(self.get_cc_table())
        print(self.get_summary())

    def get_ic_table(self) -> Table:
        """Returns a Rich table with information about the instance classes."""
        if self.is_infeasible_sol():
            return Table(
                title=f"Non feasible solution. [bold red]{self.sol.solving_stats.ilp_status}"
            )

        table = Table(
            "IC",
            Column(header="Cost", justify="right"),
            title="IC allocation (only used ICs)",
        )

        alloc = self.sol.alloc

        total_num_ics = 0
        total_cost = 0.0
        for ic, ic_allocations in alloc.ics.items():
            total_num_ics += ic_allocations
            cost = (ic.price * self.sol.problem.sched_time_size).to_reduced_units()
            total_cost += cost * ic_allocations

            table.add_row(f"{ic.name}[{ic_allocations}]", f"{cost * ic_allocations:.6f}")

        table.add_section()

        table.add_row(
            f"total: {total_num_ics}",
            f"{total_cost:.6f}",
        )

        return table

    def get_cc_table(self) -> Table:
        """Returns a Rich table with information about the container classes."""
        if self.is_infeasible_sol():
            return Table(
                title=f"Non feasible solution. [bold red]{self.sol.solving_stats.ilp_status}"
            )

        alloc = self.sol.alloc

        table = Table(
            "IC",
            "Container",
            "App",
            Column(header="Perf", justify="right"),
            title="Container allocation (only used ICs)",
        )
        for app in self.sol.problem.system.apps:
            total_num_ics = 0
            total_num_replicas = 0
            total_perf = Requests("0 req")
            for container, replicas in alloc.containers.items():
                if container.app.name != app.name or replicas == 0:
                    continue
                app = container.app
                total_num_replicas += 1
                total_num_ics += 1
                perf_cc = self.sol.problem.system.perfs[(container.ic, app)]
                perf_cc = (perf_cc * self.sol.problem.sched_time_size).to_reduced_units()
                total_perf += perf_cc * replicas

                table.add_row(
                    container.ic.name,
                    f"{container.norm_name} (x{int(replicas)})",
                    app.name,
                    f"{perf_cc} (x{int(replicas)})",
                )
            table.add_row(
                f"total: {total_num_ics}",
                f"{int(total_num_replicas)}",
                "",
                f"{total_perf:.3f}",
            )
            table.add_section()

        return table

    def is_infeasible_sol(self):
        """Returns True if the solution is infeasible."""
        return self.sol.solving_stats.ilp_status not in [
            FcmaStatus.OPTIMAL,
            FcmaStatus.FEASIBLE,
        ]


class ProblemPrettyPrinter:
    """Utility functions to show pretty presentation of a problem."""

    def __init__(self, problem: Problem) -> None:
        self.problem: Problem = problem

    def print(self) -> None:
        """Prints information about the problem."""
        self.print_ics()
        self.print_ccs()
        self.print_apps()
        self.print_perfs()

    def table_ics(self) -> Table:
        """Returns a table with information about the instance classes."""
        table = Table(title="Instance classes")
        table.add_column("Instance class")
        table.add_column("Cores", justify="right")
        table.add_column("Mem", justify="right")
        table.add_column("Price", justify="right")

        for ic in self.problem.system.ics:
            table.add_row(
                ic.name, str(ic.cores), str(ic.mem), str(ic.price)
            )

        return table

    def print_ics(self) -> None:
        """Prints information about the instance classes."""
        print(self.table_ics())

    def table_ccs(self) -> Table:
        """Returns a table with information about the container classes."""
        table = Table(title="Container classes")
        table.add_column("Container class")
        table.add_column("Cores", justify="right")
        table.add_column("Mem", justify="right")

        for cc in self.problem.system.ccs:
            table.add_row(cc.name, str(cc.cores), str(cc.mem))

        return table

    def print_ccs(self) -> None:
        """Prints information about the container classes."""
        print(self.table_ccs())

    def table_apps(self) -> Table:
        """Returns a rich table with information about the apps, including the
        workload"""
        table = Table(title="Apps")
        table.add_column("Name")
        table.add_column("Workload", justify="right")

        for app in self.problem.system.apps:
            wl = self.problem.workloads[app]
            table.add_row(app.name, str(wl.num_reqs / wl.time_slot_size))

        return table

    def print_apps(self) -> None:
        """Prints information about the apps."""
        print(self.table_apps())

    def print_perfs(self) -> None:
        """Prints information about the performance."""
        table = Table(title="Performances")
        table.add_column("Instance class")
        table.add_column("Container class")
        table.add_column("App")
        table.add_column("RPS", justify="right")
        table.add_column("Price per million req.", justify="right")

        for ic in self.problem.system.ics:
            first = True
            for app in self.problem.system.apps:
                for cc in self.problem.system.ccs:
                    if (ic, cc) not in self.problem.system.perfs:
                        continue  # Not all ICs handle all ccs

                    if first:
                        ic_column = f"{ic.name}"
                        first = False
                    else:
                        ic_column = ""

                    perf = self.problem.system.perfs[(ic, cc)]
                    price_per_1k_req = 1e6 * (ic.price.to("usd/h") / perf.to("req/h"))
                    table.add_row(
                        ic_column,
                        cc.name,
                        app.name,
                        str(perf.to("req/s").magnitude),
                        f"{price_per_1k_req.magnitude:.2f}",
                    )

            table.add_section()

        print(table)
