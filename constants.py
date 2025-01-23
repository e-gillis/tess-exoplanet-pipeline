#### Constants for the TLS Pipeline ####

SDE_CUTOFF = 6
MAX_PERIOD = 30
MIN_PERIOD = 0.1
TLS_THREADS = 8
MASK_METHOD = 'model' # Choose between 'remove', 'noise' and 'model'

# ARCHIVE?
ARCHIVEDIR = "tess_fits"

# FINDING LIGHTCURVES
MAX_SECTOR = 86

# VETTING CUTOFFS
SNR_VET = 3

# CORRELATION TOLERANCE
P_TOL = 0.01
DEPTH_TOL = 0.4
DUR_TOL = 0.2

# MCMC FITTING
SNR_MCMC = 6
ITERATIONS = 2000
BURN_IN = 750
