"""
Instance classes and families for the FCMA article
"""

from cloudmodel.unified.units import ComputationalUnits, CurrencyPerTime, Storage
from fcma import InstanceClass, InstanceClassFamily

# Instance class families. Firstly, the parent family and next its children
A_fm = InstanceClassFamily("A")
AC_fm = InstanceClassFamily("AC", parent_fms=A_fm)
AM_fm = InstanceClassFamily("AM", parent_fms=A_fm)
B_fm = InstanceClassFamily("B")
BC_fm = InstanceClassFamily("BC", parent_fms=B_fm)
BM_fm = InstanceClassFamily("BM", parent_fms=B_fm)

families = [A_fm, AC_fm, AM_fm, B_fm, BC_fm, BM_fm]

# Instance classes
AC1 = InstanceClass(
    name="AC1",
    price=CurrencyPerTime("0.100 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("4 gibibytes"),
    family=AC_fm,
)
AC2 = AC1.mul(2, "AC2")
AC4 = AC1.mul(4, "AC4")
AC8 = AC1.mul(8, "AC8")
AC18 = AC1.mul(18, "AC18")
AC24 = AC1.mul(24, "AC24")
AC36 = AC1.mul(36, "AC36")
AC48 = AC1.mul(48, "AC48")

AM1 = InstanceClass(
    name="AM1",
    price=CurrencyPerTime("0.140 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("16 gibibytes"),
    family=AM_fm,
)
AM2 = AM1.mul(2, "AM2")
AM4 = AM1.mul(4, "AM4")
AM8 = AM1.mul(8, "AM8")
AM18 = AM1.mul(18, "AM18")
AM24 = AM1.mul(24, "AM24")
AM36 = AM1.mul(36, "AM36")
AM48 = AM1.mul(48, "AM48")

BC1 = InstanceClass(
    name="BC1",
    price=CurrencyPerTime("0.070 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("4 gibibytes"),
    family=BC_fm,
)
BC2 = BC1.mul(2, "BC2")
BC4 = BC1.mul(4, "BC4")
BC8 = BC1.mul(8, "BC8")
BC18 = BC1.mul(18, "BC18")
BC24 = BC1.mul(24, "BC24")
BC36 = BC1.mul(36, "BC36")
BC48 = BC1.mul(48, "BC48")

BM1 = InstanceClass(
    name="BM1",
    price=CurrencyPerTime("0.110 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("16 gibibytes"),
    family=BM_fm,
)
BM2 = BM1.mul(2, "BM2")
BM4 = BM1.mul(4, "BM4")
BM8 = BM1.mul(8, "BM8")
BM18 = BM1.mul(18, "BM18")
BM24 = BM1.mul(24, "BM24")
BM36 = BM1.mul(36, "BM36")
BM48 = BM1.mul(48, "BM48")
