"""
Instance classes and families for the FCMA article
"""

from cloudmodel.unified.units import ComputationalUnits, CurrencyPerTime, Storage
from fcma import InstanceClass, InstanceClassFamily

# Instance class families. Firstly, the parent family and next its children
A_fm = InstanceClassFamily("A")
A_fm = InstanceClassFamily("A", parent_fms=A_fm)
Am_fm = InstanceClassFamily("Am", parent_fms=A_fm)
B_fm = InstanceClassFamily("B")
B_fm = InstanceClassFamily("B", parent_fms=B_fm)
Bm_fm = InstanceClassFamily("Bm", parent_fms=B_fm)

families = [A_fm, A_fm, Am_fm, B_fm, B_fm, Bm_fm]

# Instance classes
A_1core_4GB = InstanceClass(
    name="A_1core_4GB",
    price=CurrencyPerTime("0.100 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("4 gibibytes"),
    family=A_fm,
)
A_2core_8GB = A_1core_4GB.mul(2, "A_2core_4GB")
A_4core_16GB = A_1core_4GB.mul(4, "A_4core_16GB")
A_8core_32GB = A_1core_4GB.mul(8, "A_8core_32GB")
A_18core_72GB = A_1core_4GB.mul(18, "A_18core_72GB")
A_24core_96GB = A_1core_4GB.mul(24, "A_24core_96GB")
A_36core_144GB = A_1core_4GB.mul(36, "A_36core_144GB")
A_48core_192GB = A_1core_4GB.mul(48, "A_48core_192GB")

Am_1core_16GB = InstanceClass(
    name="Am_1core_16GB",
    price=CurrencyPerTime("0.140 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("16 gibibytes"),
    family=Am_fm,
)
Am_2core_32GB = A_1core_4GB.mul(2, "Am_2core_32GB")
Am_4core_64GB = A_1core_4GB.mul(4, "Am_4core_64GB")
Am_8core_128GB = A_1core_4GB.mul(8, "Am_8core_128GB")
Am_18core_288GB = A_1core_4GB.mul(18, "Am_18core_288GB")
Am_24core_384GB = A_1core_4GB.mul(24, "Am_24core_384GB")
Am_36core_576GB = A_1core_4GB.mul(36, "Am_36core_576GB")
Am_48core_768GB = A_1core_4GB.mul(48, "Am_48core_768GB")

B_1core_4GB = InstanceClass(
    name="B_1core_4GB",
    price=CurrencyPerTime("0.070 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("4 gibibytes"),
    family=B_fm,
)
B_2core_8GB = B_1core_4GB.mul(2, "B_2core_4GB")
B_4core_16GB = B_1core_4GB.mul(4, "B_4core_16GB")
B_8core_32GB = B_1core_4GB.mul(8, "B_8core_32GB")
B_18core_72GB = B_1core_4GB.mul(18, "B_18core_72GB")
B_24core_96GB = B_1core_4GB.mul(24, "B_24core_96GB")
B_36core_144GB = B_1core_4GB.mul(36, "B_36core_144GB")
B_48core_192GB = B_1core_4GB.mul(48, "B_48core_192GB")

Bm_1core_16GB = InstanceClass(
    name="Bm_1core_16GB",
    price=CurrencyPerTime("0.110 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("16 gibibytes"),
    family=Bm_fm,
)
Bm_2core_32GB = B_1core_4GB.mul(2, "Bm_2core_32GB")
Bm_4core_64GB = B_1core_4GB.mul(4, "Bm_4core_64GB")
Bm_8core_128GB = B_1core_4GB.mul(8, "Bm_8core_128GB")
Bm_18core_288GB = B_1core_4GB.mul(18, "Bm_18core_288GB")
Bm_24core_384GB = B_1core_4GB.mul(24, "Bm_24core_384GB")
Bm_36core_576GB = B_1core_4GB.mul(36, "Bm_36core_576GB")
Bm_48core_768GB = B_1core_4GB.mul(48, "Bm_48core_768GB")
