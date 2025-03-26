import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler

# TERMINAL EXECUTION COMMAND
# sudo /path/to/your/virtualenv/bin/python path/to/your/project_folder/experiment.py


# --------------------------------------------------START OF VARIABLES--------------------------------------------------#

# FIGURE TEMPLATE
params = {
    "font.size": 15,
    "axes.labelsize": 18,
    "axes.grid": True,
    "grid.linestyle": "dashed",
    "grid.alpha": 0.5,
    "scatter.marker": "x",
}
plt.style.use("tableau-colorblind10")
plt.rc(
    "axes",
    prop_cycle=(
        plt.rcParams["axes.prop_cycle"]
        + cycler("linestyle", ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"])
    ),
)

texture_1 = {"hatch": "/"}
texture_2 = {"hatch": "."}
texture_3 = {"hatch": "\\"}
texture_4 = {"hatch": "x"}
texture_5 = {"hatch": "o"}
plt.rcParams.update(params)

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

size_db = [1000, 2000, 3000, 4000, 5000]

df = pd.read_csv(
    "emissions_variable_db_size_plaintext.csv",
    usecols=["emissions", "energy_consumed", "duration"],
)
emissions_plaintext = df["emissions"].to_list()
energy_plaintext = df["energy_consumed"].to_list()
plaintext_per_query_energy = [
    energy_plaintext[i] / 1000 for i in range(len(energy_plaintext))
]
plaintext_per_query_emissions = [
    emissions_plaintext[i] / 1000 for i in range(len(emissions_plaintext))
]

df = pd.read_csv(
    "emissions_variable_db_size_swissse.csv",
    usecols=["emissions", "energy_consumed", "duration"],
)
emissions_swissse = df["emissions"].to_list()
energy_swissse = df["energy_consumed"].to_list()
swissse_per_query_emissions = [
    emissions_swissse[i] / 1000 for i in range(len(emissions_swissse))
]
swissse_per_query_energy = [
    energy_swissse[i] / 1000 for i in range(len(energy_swissse))
]

fig, ax = plt.subplots()
ax.plot(size_db, energy_plaintext, label="Plaintext database", marker="x")
ax.plot(size_db, energy_swissse, label="Encrypted database", marker="o")
ax.set(xlabel="Database size", ylabel="Average Energy Consumption\n per query (kWh)")
ax.legend(prop={"size": 12}, framealpha=0.80)
ax.set_axisbelow(True)
ax.yaxis.grid(color="gray", linestyle="dashed")
ax.set_yscale("log")
fig.tight_layout()
fig.savefig("comparison_energy_db_size.png", dpi=400)

fig, ax = plt.subplots()
ax.plot(size_db, emissions_plaintext, label="Plaintext database", marker="x")
ax.plot(size_db, emissions_swissse, label="Encrypted database", marker="o")
ax.set(xlabel="Database size", ylabel="Average Carbon footprint\n per query (kgCO₂eq)")
ax.legend(prop={"size": 12}, framealpha=0.80)
ax.set_axisbelow(True)
ax.yaxis.grid(color="gray", linestyle="dashed")
ax.set_yscale("log")
fig.tight_layout()
fig.savefig("comparison_emissions_db_size.png", dpi=400)
