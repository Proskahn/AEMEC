import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 160, 501)  # Time, s
I = 1.3 + t/160     # Current density, A/cm²

plt.plot(t, I, linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel(r"Cell Voltage (V)")
plt.title(r"$I(t)=1.3 + \frac{t}{160}\ \mathrm{V}$")
plt.xlim(0, 160)
plt.ylim(1.3, 2.5)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("voltage_vs_time.png", dpi=300)
plt.show()