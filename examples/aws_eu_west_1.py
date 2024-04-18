"""
AWS instance classes and families for region eu-west-1 (Ireland)
Important. This file is an example and so prices and instances do not have to agree with the real ones.
"""

from cloudmodel.unified.units import ComputationalUnits, CurrencyPerTime, Storage
from fcma import InstanceClass, InstanceClassFamily

# Instance class families. Firstly, the parent family and next its children
c5_m5_r5_fm = InstanceClassFamily("c5_m5_r5")
c5_fm = InstanceClassFamily("c5", parent_fms=c5_m5_r5_fm)
m5_fm = InstanceClassFamily("m5", parent_fms=c5_m5_r5_fm)
r5_fm = InstanceClassFamily("r5", parent_fms=c5_m5_r5_fm)

# Parent families could be also defined after children families
c6g_fm = InstanceClassFamily("c6g")
m6g_fm = InstanceClassFamily("m6g")
r6g_fm = InstanceClassFamily("r6g")
c6g_m6g_r6g_fm = InstanceClassFamily("c6g_m6g_r6g")
c6g_fm.add_parent_families(c6g_m6g_r6g_fm)
m6g_fm.add_parent_families(c6g_m6g_r6g_fm)
r6g_fm.add_parent_families(c6g_m6g_r6g_fm)

c6i_m6i_r6i_fm = InstanceClassFamily("c6i_m6i_r6i")
c6i_fm = InstanceClassFamily("c6i", parent_fms=c6i_m6i_r6i_fm)
m6i_fm = InstanceClassFamily("m6i", parent_fms=c6i_m6i_r6i_fm)
r6i_fm = InstanceClassFamily("r6i", parent_fms=c6i_m6i_r6i_fm)

c7a_m7a_r7a_fm = InstanceClassFamily("c7a_m7a_r7a")
c7a_fm = InstanceClassFamily("c7a", parent_fms=c7a_m7a_r7a_fm)
m7a_fm = InstanceClassFamily("m7a", parent_fms=c7a_m7a_r7a_fm)
r7a_fm = InstanceClassFamily("r7a", parent_fms=c7a_m7a_r7a_fm)

families = [
    c5_m5_r5_fm,
    c5_fm,
    m5_fm,
    r5_fm,
    c6g_m6g_r6g_fm,
    c6g_fm,
    m6g_fm,
    r6g_fm,
    c6i_m6i_r6i_fm,
    c6i_fm,
    m6i_fm,
    r6i_fm,
    c7a_m7a_r7a_fm,
    c7a_fm,
    m7a_fm,
    r7a_fm,
]

# Instance classes
c5_large = InstanceClass(
    name="c5.large",
    price=CurrencyPerTime("0.096 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("4 gibibytes"),
    family=c5_fm,
)
c5_xlarge = c5_large.mul(2, "c5.xlarge")
c5_2xlarge = c5_xlarge.mul(2, "c5.2xlarge")
c5_4xlarge = c5_xlarge.mul(4, "c5.4xlarge")
c5_9xlarge = c5_xlarge.mul(9, "c5.9xlarge")
c5_12xlarge = c5_xlarge.mul(12, "c5.12xlarge")
c5_18xlarge = c5_xlarge.mul(18, "c5.18xlarge")
c5_24xlarge = c5_xlarge.mul(24, "c5.24xlarge")

m5_large = InstanceClass(
    name="m5.large",
    price=CurrencyPerTime("0.107 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("8 gibibytes"),
    family=m5_fm,
)
m5_xlarge = m5_large.mul(2, "m5.xlarge")
m5_2xlarge = m5_xlarge.mul(2, "m5.2xlarge")
m5_4xlarge = m5_xlarge.mul(4, "m5.4xlarge")
m5_9xlarge = m5_xlarge.mul(9, "m5.9xlarge")
m5_12xlarge = m5_xlarge.mul(12, "m5.12xlarge")
m5_18xlarge = m5_xlarge.mul(18, "m5.18xlarge")
m5_24xlarge = m5_xlarge.mul(24, "m5.24xlarge")

r5_large = InstanceClass(
    name="r5.large",
    price=CurrencyPerTime("0.141 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("16 gibibytes"),
    family=r5_fm,
)
r5_xlarge = r5_large.mul(2, "r5.xlarge")
r5_2xlarge = r5_xlarge.mul(2, "r5.2xlarge")
r5_4xlarge = r5_xlarge.mul(4, "r5.4xlarge")
r5_9xlarge = r5_xlarge.mul(9, "r5.9xlarge")
r5_12xlarge = r5_xlarge.mul(12, "r5.12xlarge")
r5_18xlarge = r5_xlarge.mul(18, "r5.18xlarge")
r5_24xlarge = r5_xlarge.mul(24, "r5.24xlarge")

c6g_large = InstanceClass(
    name="c6g.large",
    price=CurrencyPerTime("0.073 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("4 gibibytes"),
    family=c6g_fm,
)
c6g_xlarge = c6g_large.mul(2, "c6g.xlarge")
c6g_2xlarge = c6g_xlarge.mul(2, "c6g.2xlarge")
c6g_4xlarge = c6g_xlarge.mul(4, "c6g.4xlarge")
c6g_9xlarge = c6g_xlarge.mul(9, "c6g.9xlarge")
c6g_12xlarge = c6g_xlarge.mul(12, "c6g.12xlarge")
c6g_18xlarge = c6g_xlarge.mul(18, "c6g.18xlarge")
c6g_24xlarge = c6g_xlarge.mul(24, "c6g.24xlarge")

m6g_large = InstanceClass(
    name="m6g.large",
    price=CurrencyPerTime("0.086 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("8 gibibytes"),
    family=m6g_fm,
)
m6g_xlarge = m6g_large.mul(2, "m6g.xlarge")
m6g_2xlarge = m6g_xlarge.mul(2, "m6g.2xlarge")
m6g_4xlarge = m6g_xlarge.mul(4, "m6g.4xlarge")
m6g_9xlarge = m6g_xlarge.mul(9, "m6g.9xlarge")
m6g_12xlarge = m6g_xlarge.mul(12, "m6g.12xlarge")
m6g_18xlarge = m6g_xlarge.mul(18, "m6g.18xlarge")
m6g_24xlarge = m6g_xlarge.mul(24, "m6g.24xlarge")

r6g_large = InstanceClass(
    name="r6g.large",
    price=CurrencyPerTime("0.1128 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("16 gibibytes"),
    family=r6g_fm,
)
r6g_xlarge = r6g_large.mul(2, "r6g.xlarge")
r6g_2xlarge = r6g_xlarge.mul(2, "r6g.2xlarge")
r6g_4xlarge = r6g_xlarge.mul(4, "r6g.4xlarge")
r6g_9xlarge = r6g_xlarge.mul(9, "r6g.9xlarge")
r6g_12xlarge = r6g_xlarge.mul(12, "r6g.12xlarge")
r6g_18xlarge = r6g_xlarge.mul(18, "r6g.18xlarge")
r6g_24xlarge = r6g_xlarge.mul(24, "r6g.24xlarge")

c6i_large = InstanceClass(
    name="c6i.large",
    price=CurrencyPerTime("0.0912 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("4 gibibytes"),
    family=c6i_fm,
)
c6i_xlarge = c6i_large.mul(2, "c6i.xlarge")
c6i_2xlarge = c6i_xlarge.mul(2, "c6i.2xlarge")
c6i_4xlarge = c6i_xlarge.mul(4, "c6i.4xlarge")
c6i_8xlarge = c6i_xlarge.mul(8, "c6i.8xlarge")
c6i_12xlarge = c6i_xlarge.mul(12, "c6i.12xlarge")
c6i_16xlarge = c6i_xlarge.mul(16, "c6i.16xlarge")
c6i_24xlarge = c6i_xlarge.mul(24, "c6i.24xlarge")
c6i_32xlarge = c6i_xlarge.mul(32, "c6i.32xlarge")

m6i_large = InstanceClass(
    name="m6i.large",
    price=CurrencyPerTime("0.107 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("8 gibibytes"),
    family=c6i_fm,
)
m6i_xlarge = m6i_large.mul(2, "m6i.xlarge")
m6i_2xlarge = m6i_xlarge.mul(2, "m6i.2xlarge")
m6i_4xlarge = m6i_xlarge.mul(4, "m6i.4xlarge")
m6i_8xlarge = m6i_xlarge.mul(8, "m6i.8xlarge")
m6i_12xlarge = m6i_xlarge.mul(12, "m6i.12xlarge")
m6i_16xlarge = m6i_xlarge.mul(16, "m6i.16xlarge")
m6i_24xlarge = m6i_xlarge.mul(24, "m6i.24xlarge")
m6i_32xlarge = m6i_xlarge.mul(32, "m6i.32xlarge")

r6i_large = InstanceClass(
    name="r6i.large",
    price=CurrencyPerTime("0.141 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("16 gibibytes"),
    family=c6i_fm,
)
r6i_xlarge = r6i_large.mul(2, "r6i.xlarge")
r6i_2xlarge = r6i_xlarge.mul(2, "r6i.2xlarge")
r6i_4xlarge = r6i_xlarge.mul(4, "r6i.4xlarge")
r6i_8xlarge = r6i_xlarge.mul(8, "r6i.8xlarge")
r6i_12xlarge = r6i_xlarge.mul(12, "r6i.12xlarge")
r6i_16xlarge = r6i_xlarge.mul(12, "r6i.16xlarge")
r6i_24xlarge = r6i_xlarge.mul(24, "r6i.24xlarge")
r6i_32xlarge = r6i_xlarge.mul(32, "r6i.32xlarge")

c7a_large = InstanceClass(
    name="c7a.large",
    price=CurrencyPerTime("0.11012 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("4 gibibytes"),
    family=c7a_fm,
)
c7a_xlarge = c7a_large.mul(2, "c7a.xlarge")
c7a_2xlarge = c7a_xlarge.mul(2, "c7a.2xlarge")
c7a_4xlarge = c7a_xlarge.mul(4, "c7a.4xlarge")
c7a_8xlarge = c7a_xlarge.mul(8, "c7a.8xlarge")
c7a_12xlarge = c7a_xlarge.mul(12, "c7a.12xlarge")
c7a_16xlarge = c7a_xlarge.mul(16, "c7a.16xlarge")
c7a_24xlarge = c7a_xlarge.mul(24, "c7a.24xlarge")
c7a_32xlarge = c7a_xlarge.mul(32, "c7a.32xlarge")
c7a_48xlarge = c7a_xlarge.mul(48, "c7a.48xlarge")

m7a_large = InstanceClass(
    name="m7a.large",
    price=CurrencyPerTime("0.1292 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("8 gibibytes"),
    family=m7a_fm,
)
m7a_xlarge = m7a_large.mul(2, "m7a.xlarge")
m7a_2xlarge = m7a_xlarge.mul(2, "m7a.2xlarge")
m7a_4xlarge = m7a_xlarge.mul(4, "m7a.4xlarge")
m7a_8xlarge = m7a_xlarge.mul(8, "m7a.8xlarge")
m7a_12xlarge = m7a_xlarge.mul(12, "m7a.12xlarge")
m7a_16xlarge = m7a_xlarge.mul(16, "m7a.16xlarge")
m7a_24xlarge = m7a_xlarge.mul(24, "m7a.24xlarge")
m7a_32xlarge = m7a_xlarge.mul(32, "m7a.32xlarge")
m7a_48xlarge = m7a_xlarge.mul(48, "m7a.48xlarge")

r7a_large = InstanceClass(
    name="r7a.large",
    price=CurrencyPerTime("0.17026 usd/hour"),
    cores=ComputationalUnits("1 cores"),
    mem=Storage("16 gibibytes"),
    family=r7a_fm,
)
r7a_xlarge = r7a_large.mul(2, "r7a.xlarge")
r7a_2xlarge = r7a_xlarge.mul(2, "r7a.2xlarge")
r7a_4xlarge = r7a_xlarge.mul(4, "r7a.4xlarge")
r7a_8xlarge = r7a_xlarge.mul(8, "r7a.8xlarge")
r7a_12xlarge = r7a_xlarge.mul(12, "r7a.12xlarge")
r7a_16xlarge = r7a_xlarge.mul(16, "r7a.16xlarge")
r7a_24xlarge = r7a_xlarge.mul(24, "r7a.24xlarge")
r7a_32xlarge = r7a_xlarge.mul(32, "r7a.32xlarge")
r7a_48xlarge = r7a_xlarge.mul(48, "r7a.48xlarge")
