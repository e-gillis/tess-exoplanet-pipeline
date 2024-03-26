import numpy as np
import requests, re, os
from astropy.io import fits
from astroquery.mast import Catalogs
import detrending_modules as dt


from misc_functions import *


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
    radius_err = result[0][71]
    mass = result[0][72]
    mass_err = result[0][73]
    
    RA = result[0][118]
    Dec = result[0][119]
    
    return Teff, logg, radius, radius_err, mass, mass_err, RA, Dec


def get_tess_data(tic, minsector=1, mask_flares=True, maxsector=65, sigclip=True):
    """Return TESS timeseries arrays based on the tic
    
    === Arguments ===
    tic: int
        TESS identifier number for the target
    minsector: int
        Minimum sector to retrieve data from. Default: 1
    maxsector: int
        Maximum sector to retrieve data from. Default: 65
        
    === Returns ===
    bjd: numpy array    
        BJD of each exposure
    fnorm: numpy array
        Normalized PDSCAP flux recoded by TESS
    efnorm: numpy array
        Error on the flux recorded by TESS
    sectors: numpy array
        Sector in which each exposure was taken
    qual_flags: numpy array
        Quality flag for each exposure taken
    texps: numpy array
        Exposure time for each exposure. Should be 2 minutes
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
        
        # Add exposure time
        hdr = hdus[1].header
        texps = np.repeat(hdr["FRAMETIM"]*hdr["NUM_FRM"]/(60*60*24), ftmp.size)
        
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

    # Cut down the data        
    cut = np.ones(len(bjd), dtype=bool)
    cut = cut & (qual_flags == 0)
    cut = cut & (np.isfinite(fnorm))
    cut = cut & (np.isfinite(efnorm)) 
    cut = cut & (np.isfinite(bjd))

    bjd, fnorm, efnorm = bjd[cut], fnorm[cut], efnorm[cut], 
    sectors, qual_flags, texps = sectors[cut], qual_flags[cut], texps[cut]

    # Now sigma clip and mask flares
    cut = np.ones(len(bjd), dtype=bool)
    if sigclip:
        cut = upper_sigma_clip(fnorm, sig=7)  & cut
    if mask_flares:
        cut = dt.mask_flares(fnorm, bjd, width=100) & cut
    
    bjd, fnorm, efnorm = bjd[cut], fnorm[cut], efnorm[cut], 
    sectors, qual_flags, texps = sectors[cut], qual_flags[cut], texps[cut]
        
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


