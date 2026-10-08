import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
    "font.serif": "Helvetica",
})


arr = np.loadtxt('Mstellar_table.txt', skiprows=1)

fig = plt.figure(figsize=(10,5))
plt.subplots_adjust(hspace=0.15, wspace=0.20) # h=height
gs = fig.add_gridspec(1, 1,
                      height_ratios=[1],
                      width_ratios=[1])
ax = fig.add_subplot(gs[0, 0])

ax.errorbar(arr[:,0], arr[:,4], yerr=[arr[:,3],arr[:,5]], 
             fmt=' ', capsize=5, capthick=1, elinewidth=1, 
             marker='o', markersize=6, color="#C82423", alpha=1.0, 
             zorder=2)
ax.text(0.01, 0.01, 
         f'Avg error = {np.mean((arr[:,3]+arr[:,5])/2):.2f}', 
         fontsize=12, color='brown', ha='left', va='bottom', 
         transform=ax.transAxes)

ax.set_ylabel(r'$\log_{10}{M*}$')
ax.set_xlabel( 'Slit #')
ax.minorticks_on()
ax.grid(alpha=0.75, linestyle='--', zorder=-1)
                      
plt.savefig("Mstellar_distribution.jpg", dpi=150, bbox_inches='tight')