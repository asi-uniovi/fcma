"""
AWS instance classes and families for region eu-west-1 (Irland)
Important. This file is an example and so prices and instances do not have to agree with the real ones.
"""
from fcma import InstanceClass, InstanceClassFamily
from cloudmodel.unified.units import (ComputationalUnits, CurrencyPerTime, Storage)

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

families = [
        c5_m5_r5_fm, c5_fm, m5_fm, r5_fm,
        c6g_m6g_r6g_fm, c6g_fm, m6g_fm, r6g_fm
]

# Instance classes
c5_large = InstanceClass(
        name="c5.large",
        price=CurrencyPerTime("0.085 usd/hour"),
        cores=ComputationalUnits("1 cores"),
        mem=Storage("4 gibibytes"),
        family=c5_fm
)
c5_xlarge = (c5_large * 2).set_name("c5.xlarge")
c5_2xlarge = (c5_xlarge * 2).set_name("c5.2xlarge")
c5_4xlarge = (c5_xlarge * 4).set_name("c5.4xlarge")
c5_9xlarge = (c5_xlarge * 9).set_name("c5.9xlarge")
c5_12xlarge = (c5_xlarge * 12).set_name("c5.12xlarge")
c5_18xlarge = (c5_xlarge * 18).set_name("c5.18xlarge")
c5_24xlarge = (c5_xlarge * 24).set_name("c5.24xlarge")

m5_large = InstanceClass(
        name="m5.large",
        price=CurrencyPerTime("0.096 usd/hour"),
        cores=ComputationalUnits("1 cores"),
        mem=Storage("8 gibibytes"),
        family = m5_fm
)
m5_xlarge = (m5_large * 2).set_name("m5.xlarge")
m5_2xlarge = (m5_xlarge * 2).set_name("m5.2xlarge")
m5_4xlarge = (m5_xlarge * 4).set_name("m5.4xlarge")
m5_9xlarge = (m5_xlarge * 9).set_name("m5.9xlarge")
m5_12xlarge = (m5_xlarge * 12).set_name("m5.12xlarge")
m5_18xlarge = (m5_xlarge * 18).set_name("m5.18xlarge")
m5_24xlarge = (m5_xlarge * 24).set_name("m5.24xlarge")

r5_large = InstanceClass(
        name="r5.large",
        price=CurrencyPerTime("0.126 usd/hour"),
        cores=ComputationalUnits("1 cores"),
        mem=Storage("16 gibibytes"),
        family=r5_fm
)
r5_xlarge = (r5_large * 2).set_name("r5.xlarge")
r5_2xlarge = (r5_xlarge * 2).set_name("r5.2xlarge")
r5_4xlarge = (r5_xlarge * 4).set_name("r5.4xlarge")
r5_9xlarge = (r5_xlarge * 9).set_name("r5.9xlarge")
r5_12xlarge = (r5_xlarge * 12).set_name("r5.12xlarge")
r5_18xlarge = (r5_xlarge * 18).set_name("r5.18xlarge")
r5_24xlarge = (r5_xlarge * 24).set_name("r5.24xlarge")

c6g_large = InstanceClass(
        name="c6g.large",
        price=CurrencyPerTime("0.068 usd/hour"),
        cores=ComputationalUnits("1 cores"),
        mem=Storage("4 gibibytes"),
        family=c6g_fm
)
c6g_xlarge = (c6g_large * 2).set_name("c6g.xlarge")
c6g_2xlarge = (c6g_xlarge * 2).set_name("c6g.2xlarge")
c6g_4xlarge = (c6g_xlarge * 4).set_name("c6g.4xlarge")
c6g_9xlarge = (c6g_xlarge * 9).set_name("c6g.9xlarge")
c6g_12xlarge = (c6g_xlarge * 12).set_name("c6g.12xlarge")
c6g_18xlarge = (c6g_xlarge * 18).set_name("c6g.18xlarge")
c6g_24xlarge = (c6g_xlarge * 24).set_name("c6g.24xlarge")

m6g_large = InstanceClass(
        name="c6g.large",
        price=CurrencyPerTime("0.0896 usd/hour"),
        cores=ComputationalUnits("1 cores"),
        mem=Storage("4 gibibytes"),
        family=m6g_fm
)
m6g_xlarge = (m6g_large * 2).set_name("m6g.xlarge")
m6g_2xlarge = (m6g_xlarge * 2).set_name("m6g.2xlarge")
m6g_4xlarge = (m6g_xlarge * 4).set_name("m6g.4xlarge")
m6g_9xlarge = (m6g_xlarge * 9).set_name("m6g.9xlarge")
m6g_12xlarge = (m6g_xlarge * 12).set_name("m6g.12xlarge")
m6g_18xlarge = (m6g_xlarge * 18).set_name("m6g.18xlarge")
m6g_24xlarge = (m6g_xlarge * 24).set_name("m6g.24xlarge")

r6g_large = InstanceClass(
        name="r6g.large",
        price=CurrencyPerTime("0.112 usd/hour"),
        cores=ComputationalUnits("1 cores"),
        mem=Storage("4 gibibytes"),
        family=r6g_fm
)
r6g_xlarge = (r6g_large * 2).set_name("r6g.xlarge")
r6g_2xlarge = (r6g_xlarge * 2).set_name("r6g.2xlarge")
r6g_4xlarge = (r6g_xlarge * 4).set_name("r6g.4xlarge")
r6g_9xlarge = (r6g_xlarge * 9).set_name("r6g.9xlarge")
r6g_12xlarge = (r6g_xlarge * 12).set_name("r6g.12xlarge")
r6g_18xlarge = (r6g_xlarge * 18).set_name("r6g.18xlarge")
r6g_24xlarge = (r6g_xlarge * 24).set_name("r6g.24xlarge")

