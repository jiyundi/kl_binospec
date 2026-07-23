## To install, clone and change your working directory to this repo level. Run
```
pip install -e .
```

## To run, find this code
Go to `scripts` > `main_fitting.py`. This is where I start the fitting. 

If running other Python scripts, it may require extra metadata (e.g., raw imaging or spectrum data), and these files are over 1 GB, which exceeds my GitHub storage limit.

Module Architecture Diagram:
```mermaid
flowchart TB
    A[/"<b>Imaging data & MMT spectra</b><br/>(after reduction)"/] --> B[<b>visualization/*.py </b><br/>Show CCD images of MMT slit-spectra and masks.]
    A --> E
    C[/"<b>Spectroscopic redshifts</b>"/] --> E
    D[/"<b>Stellar masses</b>"/] --> E["<b>data_prep/main.py </b><br/>The pipeline to prepare PKLs. <p style="font-size: 0.75rem;"><i>Do: (1) cut and combine images & specs, (2) store galaxy & slit information, and (3) pack into one PKL for each slit. </i></p><b>Outcome: scripts/binospec_pkl/pkl/*.pkl</b>"]
    E --> F["<b>diagnostics/*.py </b><br/>Test specific PKL data and line profile functions (LPFs)."]
    
    G[/<b>config/*.yaml </b><br>Configure which parameters to fit and their prior settings.<br/>/] --> H
    E --> H[<b>scripts/main_fitting.py </b><br/>Main script to run the sampler and save results. <p style="font-size: 0.75rem;"><i>One run per slit. Use array jobs to run parallelly. </i></p><b>Outcome: scripts/Slit_*/ </b>]
    
    %%style Core Engine fill:#f0f4f8,stroke:#94a3b8,stroke-width:2px,stroke-dasharray: 5 5
    subgraph Core Engine [<b>Core fitting pipeline</b>]
        H --> I["<b>core/*.py & klm/*.py</b><br/>KL inference (model & likelihood calculation)"] --> H
    end

    H --> J[<b>summary/*.py </b><br/>Extract posteriors of each run and down-select successful runs.]
```

## Copyright hints
The folder `klm` was originally from Pranjal's `kl_measurement` [GitHub repo](https://github.com/emhuff/kl_measurement/tree/manga/klm) (ask Eric for access), and some code was modified for my preferences. Other codes are mine. Happy to take questions!
