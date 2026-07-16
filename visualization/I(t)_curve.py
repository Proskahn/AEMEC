import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 100, 501)  # Time, s
I = 0.02 * t                  # Current density, A/cm²

plt.plot(t, I, linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel(r"Current density (A/cm$^2$)")
plt.title(r"$I(t)=0.02t\ \mathrm{A/cm^2}$")
plt.xlim(0, 100)
plt.ylim(0, 2)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("current_density_vs_time.png", dpi=300)
plt.show()