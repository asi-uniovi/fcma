import pytest
from cloudmodel.unified.units import ComputationalUnits, RequestsPerTime, Storage
from fcma import App
from fcma.model import ContainerClass


@pytest.mark.parametrize(
    ("agg_level", "expected_memory"),
    [(1, 1), (2, 2), (3, 3), (4, 4)],
)
def test_container_class_interpolates_memory_by_aggregation(agg_level, expected_memory):
    container = ContainerClass(
        app=App(name="app"),
        ic=None,
        fm=None,
        cores=ComputationalUnits("1 core"),
        mem=(Storage("2 GiB"), Storage("4 GiB")),
        perf=RequestsPerTime("1 req/s"),
        aggs=(2, 4),
        agg_level=agg_level,
    )

    assert container.memv.m_as("gibibytes") == pytest.approx(expected_memory)
