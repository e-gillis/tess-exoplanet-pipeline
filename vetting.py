import numpy as np
import math
from detrending_modules import rotation_check

# Dictionary for tags
TAG_DICT = {1: 'Rotation Signal', 
            2: 'Bad Spectrum', 
            4: 'Odd Even Mismatch', 
            8: 'Duration too long', 
            16:'Low SNR',
            32:'TLS Edge'}


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
        
        snr = get_result_snr(result, lc)
        flag.append(snr < cutoff)
    
    return tag*np.array(flag)


def tls_edge(results, tag=32):
    flag = []
    
    for result in results:
        maximim, minimum = result.periods[-2], result.period[1]
        flag.append(result.period > maximum or result.period < minimum)
    
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


def correlate_results(results_list, check_T0=True):
    """Correlate results whose periods are sufficiently close
    """
    # Empty list
    correlated_results = []
    # Flattened list of all results
    flattened_results = []
    for results in results_list: 
        flattened_results.extend(results)
    
    # Keeps going until empty
    while flattened_results:
        
        # Pop the first one of flattened, start new nested list
        correlated_results.append([flattened_results.pop(0)])
        
        # Get properties to match
        base_result = correlated_results[-1][0]
        P = base_result.period
        T0 = base_result.T0
        P_delta = get_P_delta(base_result)
        depth = 1-base_result.depth
        duration = base_result.duration
        
        # Keep going until we're at the end
        i = 0
        
        while i < len(flattened_results):
            # Update P_delta to be the most generous
            P_delta = max(P_delta, get_P_delta(flattened_results[i]))
            
            # See how far off the period is
            P_ratio = max((flattened_results[i].period/P, 
                           P/flattened_results[i].period))
            period_matches = abs(round(P_ratio) - P_ratio) < 0.02*P_ratio

            # Check the depth
            depth_matches = math.isclose(depth, 
                                         1-flattened_results[i].depth,
                                         rel_tol=0.4)
            # Check the duration
            duration_matches = math.isclose(duration, 
                                            flattened_results[i].duration,
                                            rel_tol=0.2)
            
            if check_T0:
                T0_diff = abs(flattened_results[i].T0 - T0)
                P_ratio = T0_diff / P
                difference = abs(round(P_ratio) - P_ratio)
                T0_matches = difference < P_delta / (P) * P_ratio and P_ratio > 1
            else:
                T0_matches = True
                
            # Make sure everything matches
            if period_matches and depth_matches and duration_matches and T0_matches:
                correlated_results[-1].append(flattened_results.pop(i))
            else:
                i += 1
    
    return correlated_results


def get_P_delta(result):
    P_delta = result.periods[np.argmax(result.power)+1]-\
              result.periods[np.argmax(result.power)-1]
    return P_delta


def get_result_snr(result, lc):
    """Return the SNR of a transit result from the lightcurve it was derived
    from
    """
    # Length of duration in indeces
    duration_cut = int((result.duration * 60*24) / 2)
    indx = np.arange(0, len(lc.fnorm_detrend), duration_cut)

    noise_list = [np.median(lc.fnorm_detrend[indx[i]:indx[i+1]]) 
                  for i in range(len(indx) - 1)]
    median_noise = np.median(noise_list)
    # Take the MAD
    lc_noise = np.median(np.abs(np.array(noise_list) - median_noise))
    depth = 1 - result["depth"]

    # is transit count true
    snr = depth / lc_noise * (result['transit_count'])**0.5
    
    return snr


def get_tag_names(tags):
    """Get the flags associated with a list of tags, returned as a list
    """
    tag_names_list = []
    
    for tag in tags:
        tag_names_list.append(decompose_tag(tag))
    
    return tag_names_list

    
def decompose_tag(tag):
    """Given a tag, return all of the vetting criteria that tag corresponds
    to
    """
    tag_names = []
    while tag > 0:        
        # Get the largest tag represented in the tag
        largest = 2**np.floor(np.log2(tag))
        tag_names.append(TAG_DICT[largest])
        tag -= largest
        
    return tag_names
