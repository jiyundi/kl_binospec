import numpy as np

flux = np.loadtxt("raw_flux.txt").reshape(65, 65)
err  = np.loadtxt("raw_err.txt").reshape(65, 65)

good = np.isfinite(flux) & np.isfinite(err) & (err > 0)

F = np.sum(flux[good])
sigma_F = np.sqrt(np.sum(err[good]**2))
snr = F / sigma_F

print("Npix =", good.sum())
print("Flux =", F)
print("Flux uncertainty =", sigma_F)
print("SNR =", snr)