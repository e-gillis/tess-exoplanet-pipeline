import requests, re, os

import matplotlib.pyplot as plt
import numpy as np

from astropy.io import fits
from astroquery.mast import Catalogs

import pymc3 as pm
import pymc3_ext as pmx
import aesara_theano_fallback.tensor as tt
from celerite2.theano import terms, GaussianProcess
from gls import Gls

from transitleastsquares import transitleastsquares, transit_mask
from transitleastsquares import catalog_info


class TransitSearch:
    
    # Should have all star information
    def __init__(self, tic):
        
        self.tic = tic

        full_lc = TIC_LightCurve(tic)
        lc_series = full_lc.get_splits((full_lc.bjd, full_lc.fnorm, 
                                        full_lc.efnorm, full_lc.sectors, 
                                        full_lc.qual_flags, full_lc.texp))
        self.lightcurves = []
        
        for i in range(len(lc_series[0])):
            series_set = [series[i] for series in lc_series]
            self.lightcurves.append(LightCurve(*series_set))
            self.lightcurves[-1].detrend_lc()
        
        # Star Parameters
        Teff, logg, radius, radius_min, radius_max,\
        mass, mass_min, mass_max, RA, Dec = get_star_info(tic)
        
        u = catalog_info(TIC_ID=tic)[0]
        
        self.Teff = Teff
        self.logg = logg
        self.radius = radius
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.mass = mass
        self.mass_min = mass_min
        self.mass_max = mass_max
        self.RA = RA
        self.Dec = Dec
        self.u = u
        
        self.results = None
    
    
    def transit_search(self, threshold=6, max_iterations=5):
        
        self.results = []
        
        for lc in self.lightcurves:
            results_list = find_transits(lc.bjd, lc.fnorm_detrend, 
                                         period_min=1, period_max=8,
                                         show_progress_bar=False, 
                                         R_star=self.radius, M_star=self.mass,
                                         u=self.u)
            
            self.results.append(results_list)
    
    
    # Is there any point in having a continuous lightcurve?
    def vet_transits(self):
        raise NotImplementedError
    
    def plot_transits(self):
        raise NotImplementedError
        
        


class LightCurve():
    
    def __init__(self, bjd, fnorm, efnorm, sectors, qual_flags, texp):
        
        self.bjd = bjd
        self.fnorm = fnorm
        self.efnorm = efnorm
        self.sectors = sectors
        self.qual_flags = qual_flags
        self.texp = texp
        
        self.gauss_detrendeded = False
        self.median_detrendeded = False
        self.detrended = False
        
        self.Prot = []
        
        self.fnorm_detrend = None       
        
    
    def get_splits(self, series, split_bjd=False):
        # Return each dataset in series split up by sectors
        bjd_diff = self.bjd[1:] - self.bjd[:-1]
        sector_diff = np.abs(self.sectors[1:] - self.sectors[:-1])

        # Find sector boundary points
        indeces = np.arange(len(sector_diff))

        sector_jumps = indeces[sector_diff > 1] # consecutive sectors are okay
        # Do we split based on BJD?
        if split_bjd:
            bjd_jumps = indeces[bjd_diff > 10]
            jump_indeces = np.union1d(bjd_jumps, sector_jumps) + 1
        else:
            jump_indeces = sector_jumps + 1
            
        jump_indeces = np.append(np.array(0), np.append(jump_indeces, 
                                                        np.array(len(self.bjd))))

        cut_series = [[] for i in range(len(series))]

        for i in range(len(jump_indeces)-1):
            for j in range(len(series)):
                cut_series[j].append(series[j][jump_indeces[i]:jump_indeces[i+1]])

        return cut_series
    
    ### Need to work on the detrending routines
    def detrend_lc(self):
        detrended = False
        gaussian_detrended = False
        median_detrended = False
        
        try:
            gaussian_detrended = self.gaussian_detrend_lc()
        except:
            median_detrended = self.median_detrend_lc()
    
        if not gaussian_detrended:
            median_detrended = self.median_detrend_lc()
            
        self.detrended = gaussian_detrended or median_detrended
        self.gaussian_detrended = gaussian_detrended 
        self.median_detrended = median_detrended
    
    
    def gaussian_detrend_lc(self):
        
        bjd_splits, fnorm_splits, efnorm_splits =\
                self.get_splits([self.bjd, self.fnorm, self.efnorm])
        
        full_fnorm_detrend = np.zeros(len(self.fnorm))
        index = 0
        detrended = True
        
        for i in range(len(bjd_splits)):
            bjd, fnorm, efnorm = bjd_splits[i], fnorm_splits[i], efnorm_splits[i]
            residual_rotation, Prot = rotation_check(bjd, fnorm, efnorm)
            fnorm_detrend = fnorm.copy()
            count = 0
            
            while residual_rotation and count < 3:
                self.Prot.append(Prot)
                map_soln = build_model_SHO(bjd, fnorm_detrend, efnorm, Prot)
                count += 1
                fnorm_detrend -= map_soln["pred"]/1000
              
                # plt.plot(bjd, fnorm_detrend)
                # plt.show()
                
                residual_rotation, Prot = rotation_check(bjd, fnorm_detrend, efnorm)
            
            detrended = not residual_rotation and detrended
            full_fnorm_detrend[index:index+len(fnorm_detrend)] = fnorm_detrend
            index += len(fnorm_detrend)
            
        self.fnorm_detrend = full_fnorm_detrend
        self.gaussian_detrend_lc = True
        self.detrended = detrended
        
        return self.detrended

        
    
    def median_detrend_lc(self, window_length=50):
        # Find indeces corresponding to different sectors
        fnorm_splits = self.get_splits([self.fnorm])[0]
        
        # Median 
        full_fnorm_detrend = np.zeros(len(self.fnorm))
        index = 0
        detrended = True
        k = window_length // 2
        
        for i in range(len(fnorm_splits)):
            fnorm = fnorm_splits[i]
            fnorm_detrend = np.zeros(len(fnorm))
            
            for j in range(len(fnorm)):
                fnorm_slice = fnorm[max([0,j-k]):min([len(fnorm_detrend),j+k])]
                fnorm_detrend[j] = fnorm[j]/np.median(fnorm_slice)
                
            full_fnorm_detrend[index:index+len(fnorm_detrend)] = fnorm_detrend
            index += len(fnorm_detrend)
            
        self.median_detrended = True
        self.detrended = True
        self.fnorm_detrend = full_fnorm_detrend
        return self.detrended
    
    
    def plot_curve(self, series, ax_labels, show=True, savefig=None):
        bjd_start = self.bjd[0]
        series_splits = self.get_splits(series + [self.bjd])
        nrows, ncols = len(series_splits)-1, len(series_splits[0])

        fig, axs = plt.subplots(figsize=(ncols*6, nrows*3), 
                                sharex='col', sharey='row',
                                nrows=nrows, ncols=ncols, 
                                gridspec_kw={"wspace":0.02, "hspace":0.05})

        for i in range(nrows):
            axs[i][0].set_ylabel(ax_labels[i])
            for j in range(ncols):
                bjd = series_splits[-1][j] - bjd_start
                axs[i][j].scatter(bjd, series_splits[i][j], s=0.1)
                axs[i][j].set_xlim(min(bjd), max(bjd))
                axs[i][j].grid()

        for j in range(ncols):
            axs[-1][j].set_xlabel("Days since first observation")

        if show:
            plt.show()
        if savefig:
            plt.savefig(savefig)


class TIC_LightCurve(LightCurve): 
    
    def __init__(self, tic):
                
        # Make sure get tess data 
        lc_params = get_tess_data(tic)
        bjd, fnorm, efnorm, sectors, qual_flags, texp = lc_params
        
        LightCurve.__init__(self, bjd, fnorm, efnorm, sectors, qual_flags, texp)       
        
        
        
######## Helper Functions ########

        
def lnlike(y, ey, model):
    assert np.all(np.isfinite(np.ascontiguousarray(ey)))
    return -.5 * np.nansum((y-model)**2 / ey**2)


def DeltaBIC(y, ey, model, modelnull, k=5, knull=1):
    '''Transit model (from the TLS) vs the null model (i.e. a flat line)
    k is number of free parameters
    '''
    # theta = {P,T0,D,Z,baseline}
    BIC_model = k*np.log(y.size) - 2*lnlike(y, ey, model)   
    BIC_null = knull*np.log(y.size) - 2*lnlike(y, ey, modelnull)
    return BIC_model - BIC_null


# My functions

def sincurve(x, amp, T0, P, offset):
    return amp * np.sin(2*np.pi*(x-T0)/P) + offset


def rotation_check(bjd, fnorm, efnorm):
    # Should I bin the light curve?
    bjd, fnorm, efnorm = bin_curve(bjd, fnorm, efnorm)
    
    gls = Gls(((bjd, fnorm, efnorm)), fend=10, fbeg=0.1/(bjd[-1]-bjd[0]))
    
    Prot = gls.best['P']
    
    theta = gls.best['amp'], gls.best['T0']
    Prot, offset = gls.best['P'], gls.best['offset']
    model = sincurve(bjd, *theta, Prot, offset)
    model_null = np.ones(len(bjd)) * offset
    
    
    dBIC = DeltaBIC(fnorm, efnorm, model, model_null, k=4)
    
    return dBIC <= -10, Prot


def bin_curve(bjd, fnorm, efnorm, bin_width=10):
    """Bin a given lightcurve 
    """
    bin_indeces = np.arange(0, len(bjd)+bin_width, bin_width)
    
    bin_bjd    = np.zeros(len(bin_indeces)-1)
    bin_fnorm  = np.zeros(len(bin_indeces)-1)
    bin_efnorm = np.zeros(len(bin_indeces)-1)
    
    for i in range(len(bin_indeces)-1):
        j, k = bin_indeces[i], bin_indeces[i+1]
        
        bin_bjd[i] = np.mean(bjd[j:k])
        bin_fnorm[i] = np.median(fnorm[j:k])
        bin_efnorm[i] = (np.std(fnorm[j:k])**2 + np.mean(efnorm[j:k])**2)**0.5
        
    
    return bin_bjd, bin_fnorm, bin_efnorm


def build_model_SHO(bjd, fnorm, efnorm, Prot):
    fnorm = (fnorm - 1)*1000
    efnorm = efnorm*1000
    
    with pm.Model() as model:
        # Fitting a lightcurve mean
        mean = pm.Normal("mean", mu=0, sd=10)
        
        # Transit jitter and GP parameters
        # Spread of initial Data
        log_sigma_gp = pm.Uniform("log_sigma_gp", lower=-3, 
                                  upper=np.log(np.std(fnorm)))

        # Quantifying measurement uncertainty
        log_sigma_lc = pm.Normal("log_sigma_lc", 
                                 mu=np.log(np.median(efnorm)), sd=.1)
        
        # Period Fit
        log_rho_gp = pm.Normal("log_rho_gp", mu=np.log(Prot), sd=.2)
        # Damping and variation timescale
        log_tau_gp = pm.Uniform("log_tau_gp", lower=np.log(10*Prot), upper=20)

        # GP model for the light curve
        kernel = terms.SHOTerm(
            sigma=tt.exp(log_sigma_gp),
            rho=tt.exp(log_rho_gp),
            tau=tt.exp(log_tau_gp),
        )
        gp = GaussianProcess(kernel, t=bjd, yerr=tt.exp(log_sigma_lc))
        resid = fnorm
        gp.marginal("gp", observed=resid)
        pm.Deterministic("pred", gp.predict(fnorm))
        
        start = model.test_point
        map_soln = pmx.optimize(start=start,
                                vars=[log_sigma_lc, log_sigma_gp, 
                                      log_rho_gp, log_tau_gp])
        map_soln = pmx.optimize(start=map_soln, vars=[mean])
        map_soln = pmx.optimize(start=map_soln,
                                vars=[log_sigma_lc, log_sigma_gp,
                                      log_rho_gp, log_tau_gp])
        map_soln = pmx.optimize(start=map_soln)

    return map_soln


def find_transits(bjd, fnorm, threshold=6, max_iterations=5, **tls_kwargs):
    """Return a list of TLS transit candidates from a given lightcurve.
    
    === Parameters ===
    BJD: 1D numpy array
        BJD of each data point in the observation timeseries
    fnorm: 1D array
        Normalized flux at each BJD. fnorm *must* be detrended to return 
        sensible results
    threshold: float
        Threshold for a peak in SED to be considered a transit candidate
    max_iterations: int
        Maximum number of TLS iterations to run, sucessively masking each 
        significant transit candidate
    **tls_kwargs:
        Keyword arguments to pass into the tls run
        
    === Returns ===
    result_list: List[TLS result]
        List of results for transit candidates with sufficient SED, see
        transitleastsquares.readthedocs.io/en/latest/Python interface.html#return-values
        for more information
    """
    
    # All points are in the transit
    intransit = np.zeros(len(bjd), dtype=bool)
    
    # Look for thf first planet
    model = transitleastsquares(bjd, fnorm)
    result = model.power(**tls_kwargs)
    
    # Check if a planet candidate is found
    if result["SDE"] < threshold:
        return []
    
    result_list = [result]
    i = 1
    
    # Start looping finding more planets
    while i <= max_iterations:
        # Mask found transit, | is logical or, and acumulates false
        intransit = intransit | transit_mask(bjd, result_list[-1].period, 
                                             2*result_list[-1].duration, 
                                             result_list[-1].T0)
        
        # Look for planets again with transits masked
        model = transitleastsquares(bjd[~intransit], fnorm[~intransit])
        result = model.power(**tls_kwargs)
        
        # Check if planet found
        if result["SDE"] > threshold:
            result_list.append(result)
        else: # Run a grazing template to see if we missed something?
            result = model.power(transit_template='grazing',**tls_kwargs)
            if result["SDE"] > threshold:
                result_list.append(result)
            else:
                break
        # Increment
        i += 1
        
    return result_list


def get_tess_data(tic, minsector=1, maxsector=55):
    """Extract TESS data for a given tic and generate timeseries for 
    observations between minsector and maxsector. A quality cut and a sigma
    clip are performed before returning the timeseries:
    
    === Parameters === 
    tic: int or str
        TIC to retrieve data for
    minsector: int
        Minimum sector to look from
    maxsector: int
        Maximum sector to look too

    === Returns ===
    BJD: 1D numpy array
        BJD observation time of each data point
    fnorm: 1D numpy array 
        Normalized flux observed by TESS
    sectors: 1D numpy array 
        Sector which each data point was observed in
    qual_flags: 1D numpy array 
        Quality flags for each data point
    texp: 1D numpy array 
        Exposure time of each data point

    NB: Requires internet connection!
    """
    # Get all the filenames
    filenames = get_tess_filenames(tic, minsector=minsector, 
                                   maxsector=maxsector)
    # Lists to acumulate series
    bjd_list, fnorm_list, efnorm_list = [], [], []
    sectors_list, qual_flags_list, texps_list = [], [], []
    
    # Download data
    for file in filenames:
        hdus = fits.open(file)
        
        # Extract timeseries
        bjd = hdus[1].data['TIME'] + 2457000
        ftmp = hdus[1].data['PDCSAP_FLUX']
        eftmp = hdus[1].data['PDCSAP_FLUX_ERR']
        qual_flags = hdus[1].data['QUALITY']
        sectors = np.repeat(hdus[0].header['SECTOR'], ftmp.size)
        
        # Normalize
        efnorm = eftmp / np.nanmedian(ftmp)
        fnorm  = ftmp  / np.nanmedian(ftmp)
        
        hdr = hdus[1].header
        texps = np.repeat(hdr["FRAMETIM"]*hdr["NUM_FRM"]/(60*60*24), ftmp.size)
        
        # Cut down the data
        cut = (qual_flags == 0) & (np.isfinite(fnorm)) &\
              (np.isfinite(efnorm)) & (np.isfinite(bjd))
        cut = upper_sigma_clip(fnorm, sig=7, clip=cut)
        
        bjd, fnorm, efnorm = bjd[cut], fnorm[cut], efnorm[cut], 
        sectors, qual_flags, texps = sectors[cut], qual_flags[cut], texps[cut]
        
        # Acumulate the data arrays
        bjd_list.append(bjd)
        fnorm_list.append(fnorm)
        efnorm_list.append(efnorm)
        sectors_list.append(sectors)
        qual_flags_list.append(qual_flags)
        texps_list.append(texps)
        
    # Make Everything a loooong array
    bjd = np.concatenate(bjd_list)
    fnorm = np.concatenate(fnorm_list)
    efnorm = np.concatenate(efnorm_list)
    sectors = np.concatenate(sectors_list)
    qual_flags = np.concatenate(qual_flags_list)
    texps = np.concatenate(texps_list)
    
    return bjd, fnorm, efnorm, sectors, qual_flags, texps
    

def get_tess_filenames(tic, minsector=1, maxsector=55):
    """Retrive files associated with a specific TIC between minsector and 
    maxsector. This function will only retrieve the filenames of lightcurves
    with a two minute cadence hosted at:
    
    https://archive.stsci.edu/missions/tess/tid/
    
    === Parameters ===
    tic: int or str
        TIC to retrieve filenames for
    minsector: int
        Minimum sector to look from
    maxsector: int
        Maximum sector to look too
        
    === Returns ===
    filenames: list
        The paths to all 2 minute cadence lightcurves
        
    NB: Requires internet connection!
    """
    filenames = []
    url_base = 'https://archive.stsci.edu/missions/tess/tid/'
    
    # TIC is formatted into the URL in four-character chuncks
    tic_str = '{:016d}'.format(int(tic))
    tic_list = [tic_str[:4], tic_str[4:8], tic_str[8:12], tic_str[12:]]
    dir_str = ("{}/"*4).format(*tic_list)
    
    # Go through all the sectors and see if the data is there
    for j in range(minsector, maxsector+1):
        sector = f"s{j:04}/"
        page = requests.get(url_base + sector + dir_str)
                
        # 200 is a success, we have the webpage
        if page.status_code == 200:
            # Search the wevpage text for regular expression
            page_text = page.text            
            fits_names = re.findall("tess.*_lc.fits", page_text)
            # Acumulate the full path to the file
            filenames.extend([url_base + sector + dir_str + fits_name
                              for fits_name in fits_names[:1]])
        
    return filenames


def upper_sigma_clip(series, sig, clip=None):
    """
    Return a boolean array to index by to perform an upper sigma clip
    
    === Parameters ===
    series: 1D numpy array
        Array of series to be sigma clipped
    sig: float or int
        Sigma clip factor
    clip: boolean numpy array
        Starting array for the sigma clip, False data points will be masked.
        Must be the same shape as series.
        
    === Returns ===
    clip: boolean numpy array
        Boolean array of points to be clipped, same shape as series
    """
    delta = 1
    if clip is None:
        clip = np.ones(series.size, dtype=bool)
    c_size = np.sum(~clip)
    
    while delta:
        c_size = np.sum(~clip)
        mean, std = series[clip].mean(), series[clip].std()
        clip = series <= mean + std*sig
        delta = np.sum(~clip) - c_size
        
    return clip


def get_star_info(tic):
    """
    Retrieve the info of a star for a given TIC, returns a tuple of the 
    following information:
    
    (Teff, logg, radius, radius_min, radius_max, 
    mass, mass_min, mass_max, RA, Dec)
    """
    result = Catalogs.query_criteria(catalog="Tic", ID=tic).as_array()
    Teff = result[0][64]
    logg = result[0][66]
    radius = result[0][70]
    radius_max = result[0][71]
    radius_min = result[0][71]
    mass = result[0][72]
    mass_max = result[0][73]
    mass_min = result[0][73]
    
    RA = result[0][118]
    Dec = result[0][119]
    
    return Teff, logg, radius, radius_min, radius_max,\
           mass, mass_min, mass_max, RA, Dec
