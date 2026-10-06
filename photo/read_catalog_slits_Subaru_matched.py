import numpy  as np
import pandas as pd

"""
 1 id      Object ID Number
 2 RA      Right Ascension in decimal degrees
 3 Dec     Declination in decimal degrees
 4 x       x pixel coordinate
 5 y       y pixel coordinate
 6 area    Segment size
 7 B       B_Subaru band magnitude
 8 dB      Uncertainty in B
 9 V       V_Subaru band magnitude
10 dV      Uncertainty in V
11 R       R_Subaru band magnitude
12 dR      Uncertainty in R
13 I       I_Subaru band magnitude
14 dI      Uncertainty in I
15 i       i_Subaru band magnitude
16 di      Uncertainty in i
17 Z       z_Subaru band magnitude
18 dZ      Uncertainty in Z
19 zb      BPZ most likely redshift
20 zbmin   Lower limit (95% confidence)
21 zbmax   Upper limit (95% confidence)
22 tb      BPZ most likely spectal type
23 odds    P(z) contained within 2*0.06*(1+z)
24 zml     Maximum Likelihood most likely redshift
25 tml     Maximum Likelihood most likely spectral type
26 chisq   Poorness of BPZ fit: observed vs. model fluxes
27 chisq2  Modified chisq: model fluxes given error bars
28 nfdet   Number of filters in which object is detected
29 nfobs   " in which object is observed (within FOV, no bad pixels, etc.)
"""

txtname   = 'slits_Subaru_matched.txt'
mag_table = pd.read_csv(txtname, sep=" ", skiprows=0, # including this line
                        comment="#", engine="python")