"""
This module provides ways of visualizing the solutions for FCMA.
"""

from rich.table import Table, Column
from rich import print as print_rich
from cloudmodel.unified.units import CurrencyPerTime, RequestsPerTime
from .model import Solution, SolutionSummary, Vm, SolvingStats, FcmaStatus


class ProblemPrinter:
    """
    Utility methods to create pretty presentations of problems.
    """

    def __init__(self, fcma):
        self.fcma = fcma

    def print(self):
        """
        Print the problem.
        """
        print_rich(self._get_workloads_table())
        print_rich(self._get_system_table())

    def _get_workloads_table(self) -> Table:
        """
        Return a Rich table with information about the workloads.
        :return: The workload table.
        """
        table = Table("App", "Workload", title="Workloads")
        for app, workload in self.fcma._workloads.items():
            table.add_row(app.name, f"{workload:.3f}")
        return table

    def _get_system_table(self) -> Table:
        """
        Return a Rich table with information about the system.
        :return: The system table.
        """
        table = Table(
            "App",
            "Family",
            "Cores",
            "Mem",
            "Perf",
            "Aggs",
            title="System",
        )
        for (app, fm), perf in self.fcma._system.items():
            # There's a memory value per container class. If all are the same, we print only one.
            mem_set = set(i for i in perf.mem)
            if len(mem_set) == 1:
                mem = mem_set.pop().to("gibibytes").magnitude
                mem_str = f"{mem:.3f}"
            else:
                mem_str = ", ".join(f"{mem.to('gibibytes').magnitude:.3f}" for mem in perf.mem)

            table.add_row(
                app.name,
                fm.name,
                f"{perf.cores.to('cores').magnitude:.3f}",
                mem_str,
                f"{perf.perf:.3f}",
                f"{perf.aggs}",
            )
        return table


class SolutionPrinter:
    """
    Utility methods to create pretty presentations of solutions.
    """

    def __init__(self, solution: Solution):
        self._solution = solution
        self._vms = solution.allocation
        self._statistics = solution.statistics
        self.summary = SolutionSummary(solution)

    def print_containers(self):
        """
        Print solution container tables.
        """

        if self._solution.is_infeasible():
            print_rich(
                Table(title=f"Non feasible solution. [bold red]{self._statistics.final_status}")
            )
        tables = self._get_app_tables()
        keys = list(tables.keys())
        keys.sort()
        for key in keys:
            print_rich(tables[key])

    def print_vms(self):
        """
        Print solution container virtual machines.
        """

        if self._solution.is_infeasible():
            print_rich(
                Table(title=f"Non feasible solution. [bold red]{self._statistics.final_status}")
            )
        print_rich(self._get_vm_table())

    def _print_statistics(self) -> None:
        """
        Print solution statistics.
        """

        print("")
        print("Statistics")
        print("----------")
        print(f"Speed level: {self._statistics.solving_pars.speed_level}")
        if self._statistics.partial_ilp_seconds is not None:
            print(
                f"Time spent in the partial ILP problem: {self._statistics.partial_ilp_seconds:.3f} seconds"
            )
        if self._statistics.partial_ilp_status is not None:
            print(f"Solution status after the ILP problem: {self._statistics.partial_ilp_status}")
        print(f"Status previous to the allocation phase: {self._statistics.pre_allocation_status}")
        print(f"Cost before the allocation phase: {self._statistics.pre_allocation_cost:.3f}")
        print(
            f"Time spent before the allocation phase: {self._statistics.pre_allocation_seconds:.3f} seconds"
        )
        print(
            f"Time spent in the allocation phase: {self._statistics.allocation_seconds:.3f} seconds"
        )
        print(f"Final status: {self._statistics.final_status}")
        print(f"Final cost: {self._statistics.final_cost:.3f}")
        print(f"Total spent time: {self._statistics.total_seconds: .3f} seconds")
        if self._statistics.fault_tolerance_m != -1:
            print(f"Fault tolerance metric: {self._statistics.fault_tolerance_m: .3f}")
        if self._statistics.container_isolation_m != -1:
            print(f"Container isolation metric: {self._statistics.container_isolation_m: .3f}")
        if self._statistics.vm_recycling_m != -1:
            print(f"VM recycling metric: {self._statistics.vm_recycling_m: .3f}")
        if self._statistics.vm_load_balance_m != -1:
            print(f"VM load-balance metric: {self._statistics.vm_load_balance_m: .3f}")

    def print(self):
        """
        Print tables and a summary of the solution.
        """

        if self._solution.is_infeasible():
            print(f"Non feasible solution. [bold red]{self._statistics.final_status}")
            return

        self.print_vms()
        self.print_containers()
        self._print_statistics()

    def _get_vm_table(self) -> Table:
        """
        Return a Rich table with information about the virtual machines.
        :return: The virtual machines table.
        """

        table = Table("VM", Column(header="Cost", justify="right"), title="Virtual Machines")
        vm_summary = self.summary.get_vm_summary()

        for vm in vm_summary.vms:
            table.add_row(f"{vm.ic_name} (x{vm.total_num})", f"{vm.cost:.3f}")
        table.add_section()
        table.add_row(f"Total: {vm_summary.total_num}", f"{vm_summary.total_cost:.3f}")
        return table

    def _get_app_tables(self) -> dict[str, Table]:
        """
        For each application returns a Rich table with information about the allocation of its container classes.
        :return: One dictionary with table with container applications for each name.
        """

        tables: dict[str, Table] = {}
        if self._solution.is_infeasible():
            print_rich(
                Table(title=f"Non feasible solution. [bold red]{self._statistics.final_status}")
            )
            return tables

        # Get application table rows
        apps_info = self.summary.get_all_apps_allocations()
        for app_name, app_info in apps_info.items():
            table = Table(
                "VM",
                "Container",
                "App",
                "Perf",
                title=f"Container allocation for {app_name}",
            )
            for container in app_info.container_groups:
                cores = int(container.cores.m_as("mcore"))
                container_name = f"{str(container.container_name)}-{cores} mcores"
                row = (
                    container.vm_name,
                    f"{container_name} (x{container.replicas})",
                    app_name,
                    f"{container.performance.m_as('req/s'):.3f} req/s (x{container.replicas})",
                )
                table.add_row(*row)
            table.add_section()
            table.add_row(
                "total:",
                f"{app_info.total_replicas}",
                "",
                f"{app_info.total_perf.m_as('req/s'):.3f} req/s",
            )
            tables[app_name] = table
        return tables
