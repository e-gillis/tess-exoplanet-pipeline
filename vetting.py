import numpy as np
from detrending_modules import rotation_check

def rotation_signal(lc, results, tag=1):
    """Return Flag array with results whose period are too close to the star's
    rotation period flagged
    """
    # Get dominant lightcurve rotation signal
    rotation, Prot = rotation_check(lc.bjd, lc.fnorm, lc.efnorm)
    
    # See if any of the results have a period 
    if rotation:
        flag = []
        # Make sure this is positive
        for result in results:
            ratio = max([result.period/Prot, Prot/result.period])
            mod = min(ratio%1, (1-ratio)%1)
            flag.append(np.isclose(mod, 0, atol=ratio*0.02))  # 2% Tolerance?
        flag = np.array(flag)
         
        return tag*flag
    
    # Should I iterate through and mask transits?
    else:
        flag = np.array([False for result in results])
        return tag*flag
        
    
def bad_tls_spectrum(results, tag=2):
    """Return flag array with results with suspicious TLS results flagged
    """
    flag = []
    
    for result in results:
        diff = result.power[:-1]-result.power[1:]
        flag.append(sum(diff==0) / len(result.power) > 0.5)
        
    return tag*np.array(flag)


def odd_even_mismatch(results, tag=4):
    """Return flag array with results with suspicious mismatched transit depths
    flagged
    """
    mismatched = [result["odd_even_mismatch"] > 3 for result in results]
    return tag*np.array(mismatched)


def duration_period_fraction(results, tag=8):
    """Return flag array with results whose dirations are more than 10% of
    their periods flagged
    """
    dp_ratio = [result.duration/result.period > 0.1 for result in results]
    return tag*np.array(dp_ratio)


def low_snr(results, lc, cutoff=2, tag=16):
    """Return flag array with results whose SNR is low flagged
    """
    flag = []
    for result in results:
        # Length of duration in indeces
        duration_cut = int((result.duration * 60*60*24) / 2)
        indx = np.arange(0, len(lc.fnorm_detrend), duration_cut)
        
        lc_noise = np.mean([lc.fnorm_detrend[indx[i]:indx[i+1]] 
                            for i in range(len(indx) - 1)])
        
        snr = result['depth'] / lc_noise * (result['transit_count'])**0.5
        flag.append(snr < cutoff)
    
    return tag*np.array(flag)


### Vetting Helper functions ###

def cut_results(results_list, result_tags):
    """Cut nested list of results based on tag list of the same shape
    Does not support selecting specific tags yet, will be rewritten
    """
    # List to acumulate in
    cut_results_list = []
    
    # Index through
    for i in range(len(results_list)):
        res_cut = []
        for j in range(len(results_list[i])):
            # Tag check, should be more sophisticated
            if result_tags[i][j] == 0:
                res_cut.append(results_list[i][j])

        cut_results_list.append(res_cut)
        
    return cut_results_list


def correlate_results(results_list):
    """Correlate results whose periods are sufficiently close
    """
    # Empty list
    correlated_results = []
    # Flattened list of all results
    flattened_results = []
    for results in results_list: flattened_results.extend(results)
    
    # Keeps going until empty
    while flattened_results:
        
        # Pop the first one of flattened, start new nested list
        correlated_results.append([flattened_results.pop(0)])
        
        # Get the period to match
        P = correlated_results[-1][0].period
        
        # Keep going until we're at the end
        i = 0
        while i < len(flattened_results):
            # See how far off the period is
            ratio = max((flattened_results[i].period/P, 
                         P/flattened_results[i].period))
            period_matches = abs(round(ratio) - ratio) < 0.02*ratio
            # If it matches, put it in, if not go to the next one
            if period_matches:
                correlated_results[-1].append(flattened_results.pop(i))
            else:
                i += 1
    
    return correlated_results