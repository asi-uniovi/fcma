"""
This module provides ways of visualizing the solutions for FCMA.
"""

from rich.table import Table, Column
from rich import print as print_rich
from cloudmodel.unified.units import CurrencyPerTime, RequestsPerTime
from .model import Vm, SolvingStats, FcmaStatus, Solution


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

        total_num_vms = {}
        ic_prices = {}
        for fm in self._vms:
            for vm in self._vms[fm]:
                ic_name = vm.ic.name
                if ic_name not in total_num_vms:
                    total_num_vms[ic_name] = 1
                else:
                    total_num_vms[ic_name] += 1
                if ic_name not in ic_prices:
                    ic_prices[ic_name] = vm.ic.price
        total_cost = CurrencyPerTime("0 usd/hour")
        total_vms = 0
        for ic_name, total_num in total_num_vms.items():
            ic_cost = total_num * ic_prices[ic_name].to_reduced_units()
            total_cost += ic_cost
            total_vms += total_num
            table.add_row(f"{ic_name} (x{total_num})", f"{ic_cost:.3f}")
        table.add_section()
        table.add_row(f"Total: {total_vms}", f"{total_cost:.3f}")
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
        app_table_entries = {}
        for fm in self._vms:
            for vm in self._vms[fm]:
                for cg in vm.cgs:
                    cores = int(cg.cc.cores.to("mcore").magnitude)
                    container_name = f"{str(cg.cc)}-{cores} mcores"
                    app_name = cg.cc.app.name
                    row = (str(vm), container_name, cg.cc.perf, cg.replicas)
                    if app_name not in app_table_entries:
                        app_table_entries[app_name] = {
                            "rows": [row],
                            "total_perf": cg.cc.perf * cg.replicas,
                        }
                    else:
                        app_table_entries[app_name]["rows"].append(row)
                        app_table_entries[app_name]["total_perf"] += cg.cc.perf * cg.replicas

        # Get an allocation table for each application
        for app_name, app_table_entry in app_table_entries.items():
            table = Table(
                "VM",
                "Container",
                "App",
                "Perf",
                title=f"Container allocation for {app_name}",
            )
            total_app_replicas = 0
            total_app_perf = RequestsPerTime("0 req/s")
            for app_row in app_table_entry["rows"]:
                total_app_replicas += app_row[3]
                total_app_perf += app_row[2] * app_row[3]
                table.add_row(
                    app_row[0],
                    f"{app_row[1]} (x{app_row[3]})",
                    app_name,
                    f"{app_row[2].to('req/s').magnitude:.3f} req/s (x{app_row[3]})",
                )
            table.add_section()
            table.add_row("total:", f"{total_app_replicas}", "", f"{total_app_perf.magnitude:.3f} req/s")
            tables[app_name] = table

        return tables
