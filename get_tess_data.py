import numpy as np
import requests, re, os, time
from astropy.io import fits
from astroquery.mast import Catalogs, Observations
from astropy.table import Table
import detrending_modules as dt
from glob import glob
import lightkurve as lk

from misc_functions import *


def get_tess_sectors(TIC):
    """
    Using MAST, get the sectors for 2 minute cadence TESS Lightcurves
    """
    query_criteria = {"project": 'TESS', "t_exptime": [110, 130], 
                      "provenance_name": "SPOC"}
    observation = Observations.query_criteria(target_name=str(TIC), **query_criteria)   
    
    sectors = []
    for i in range(len(observation)):
        row = observation[i]
        if len(row["obs_id"].split("-"))==5 and row["dataURL"][-9:]=='s_lc.fits' and\
           (row["sequence_number"] not in sectors):
            sectors.append(row["sequence_number"])
    sectors.sort()
    
    # Copied from lightkurve
    # target_lower = f'tic {TIC}'
    # tess_match = re.match(r"^(tess|tic) ?(\d+)$", target_lower)
    # exact_target_name = f"{tess_match.group(2).zfill(9)}"
    # print(exact_target_name)
    
    return sectors



def get_star_info(tic, archivedir=None):
    """
    Retrieve the info of a star for a given TIC, returns a tuple of the 
    following information:
    
    (Teff, logg, radius, radius_min, radius_max, 
    mass, mass_min, mass_max, RA, Dec)
    """
    load = False
    tabstr = f"{archivedir}/{tic}/{tic}_dattab.ecsv"
    
    if archivedir is not None:
        make_archivedirs(tic, archivedir)
        load = os.path.isfile(tabstr)
        
    if load:
        result =  Table.read(tabstr)
        
    else:
        print("Downloading Target Data")
        retries = 0
        while retries < 3:
            try:
                result = Catalogs.query_criteria(catalog="Tic",
                         ID=tic)
                
                if archivedir is not None:
                    result.write(tabstr)
                    
                result = result.as_array()
                break
            except TimeoutError:
                time.sleep(30)
                retries += 1
        
    Teff = result[0][64]
    logg = result[0][66]
    radius = result[0][70]
    radius_err = result[0][71]
    mass = result[0][72]
    mass_err = result[0][73]
    
    RA = result[0][118]
    Dec = result[0][119]
    
    return Teff, logg, radius, radius_err, mass, mass_err, RA, Dec


def get_tess_data(tic, minsector=1, mask_flares=True, maxsector=65, sigclip=True, 
                  archivedir=None, verb=True):
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
    # Lists to acumulate series
    bjd_list, fnorm_list, efnorm_list = [], [], []
    sectors_list, qual_flags_list, texps_list = [], [], []
    
    if archivedir is not None:
        make_archivedirs(tic, archivedir)
        
        # Check if directory exists!
        if len(glob(f"{archivedir}/{tic}/*.fits")) == 0:
            # Get all the filenames
            filenames = get_tess_filenames(tic, minsector=minsector, 
                                           maxsector=maxsector, verb=verb)

            # Make a directory for the files
            for file in filenames:
                # Save the files
                hdus = fits.open(file)
                fstr = file.split("/")[-1]
                hdus.writeto(f"{archivedir}/{tic}/{fstr}")
            
        # Get all the filenames
        filenames = sorted(glob(f"{archivedir}/{tic}/*.fits"))

    else:
        # Get all the filenames
        filenames = get_tess_filenames(tic, minsector=minsector, 
                                       maxsector=maxsector, verb=verb)
    
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
    

def get_tess_filenames(tic, minsector=1, maxsector=80, max_retries=3, verb=False):
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

    if verb:
        print("Retrieving TESS Filenames")
    
    # TIC is formatted into the URL in four-character chuncks
    tic_str = '{:016d}'.format(int(tic))
    tic_list = [tic_str[:4], tic_str[4:8], tic_str[8:12], tic_str[12:]]
    dir_str = ("{}/"*4).format(*tic_list)
    
    tess_sectors = np.array(get_tess_sectors(tic))
    tess_sectors = tess_sectors[(tess_sectors >= minsector) &\
                                (tess_sectors <= maxsector)]
    
    # Go through all the sectors and see if the data is there
    for j in tess_sectors:
        sector = f"s{j:04}/"

        retry_count = 0
        while retry_count < max_retries:
            try:
                page = requests.get(url_base + sector + dir_str)
                break
            except:
                print("Connection Error, sleeping")
                time.sleep(20)
                retry_count += 1
        
        if retry_count == 3:
            continue
                
        # 200 is a success, we have the webpage
        if page.status_code == 200:
            # Search the wevpage text for regular expression
            page_text = page.text            
            fits_names = re.findall("tess.*_lc.fits", page_text)
            # Acumulate the full path to the file
            filenames.extend([url_base + sector + dir_str + fits_name
                              for fits_name in fits_names[:1]])
        
    return filenames


def make_archivedirs(tic, archivedir):
    for directory in [f"{archivedir}", f"{archivedir}/{tic}/"]:
        # See if the data has been saved before
        if not os.path.isdir(directory):
            # Make the directory
            os.mkdir(directory)
