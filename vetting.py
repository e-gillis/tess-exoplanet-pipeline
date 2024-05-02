import numpy as np
import math
from detrending_modules import rotation_check
from transitleastsquares import transit_mask
import misc_functions as misc


# Dictionary for tags
TAG_DICT = {1:  'Rotation Signal', 
            2:  'Duplicate Period', 
            4:  'Odd Even Mismatch', 
            8:  'Duration too long', 
            16: 'Low SNR',
            32: 'TLS Edge',
            64: 'LC Edge',
            128:'Deep single transit',
            256:'Inf period uncertainty',
            512:'Half sectors or more'}


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
            flag.append(np.isclose(mod, 0, atol=ratio*0.003))  # 0.3% Tolerance?
        flag = np.array(flag)
         
        return tag*flag
    
    # Should I iterate through and mask transits?
    else:
        flag = np.array([False for result in results])
        return tag*flag
        
    
def bad_tls_spectrum(results, tag=0):
    """Return flag array with results with suspicious TLS results flagged
    """
    flag = []
    
    for result in results:
        diff = result.power[:-1]-result.power[1:]
        flag.append(sum(diff==0) / len(result.power) > 0.66)
        
    return tag*np.array(flag)


def duplicate_period(results, tag=2):
    flags = np.zeros(len(results))
    
    for i, r1 in enumerate(results):
        for j, r2 in enumerate(results[i+1:]):
            if math.isclose(r1.period, r2.period, rel_tol=0.01):
                flags[j] = tag
                
    return flags
            


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


def inf_period_uncertainty(results, tag=256):
    """Return a flag array where results with infinite period uncertainty 
    are flagged
    """
    flag = [r.period_uncertainty == np.inf for r in results]
    return tag*np.array(flag)


def half_sectors_or_more(results_list, lc_lengths, tag=512):
    """Given a full set of results, flag all signals that only appear in less
    than half of the lightcurves from a target
    
    Results with a sufficiently high SDE will be excempt
    """
    full_tags = [np.zeros(i) for i in [len(r) for r in results_list]]
    
    for i, results in enumerate(results_list):
        for j, r in enumerate(results):
            # r is the base result
            if results_list[i][j].SDE > 11:
                continue
            
            count = 1
            P = r.period
            
            r_indeces = [m for m in range(len(results_list)) if m != i]
            for k in r_indeces:
                for l in range(len(results_list[k])):
                    P_ra = max(P/results_list[k][l].period,
                               results_list[k][l].period/P)
                    if math.isclose(P_ra,round(P_ra),rel_tol=0.01) and P_ra < 3:
                        count += 1
                        break
            
            # Total number of sectors where this result could have been found
            sector_num = sum(lc_lengths > P*2)
            
            if count < sector_num // 2:
                full_tags[i][j] += tag
                    
    return full_tags


# NEEDS WORK
def tls_edge(results, tag=32):
    flag = []
    
    for result in results:
        maximum, minimum = result.periods[-2], result.periods[1]
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


def old_correlate_results(results_list, ptol=0.02, durtol=0.3, depthtol=0.5, 
                          check_T0=True):
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
        # P_delta = get_P_delta(base_result)
        depth = 1-base_result.depth
        duration = base_result.duration
        
        # Keep going until we're at the end
        i = 0
        
        while i < len(flattened_results):
            # See if the two results match up
            if correlation_check(base_result, flattened_results[i], 
                                 ptol=ptol, durtol=durtol,
                                 depthtol=depthtol, check_T0=check_T0):
                correlated_results[-1].append(flattened_results.pop(i))
                
            else:
                i += 1
    
    return correlated_results
                          

def correlation_check(res1, res2, ptol=0.02, durtol=0.3, depthtol=0.5, 
                      check_T0=True):
    """Check for Correlation
    """
    # Update P_delta to be the most generous
    P_delta = max(get_P_delta(res1), get_P_delta(res2))
    # See how far off the period is
    P_ratio = max((res1.period/res2.period, 
                   res2.period/res1.period))
    period_matches = abs(round(P_ratio) - P_ratio) < ptol*P_ratio

    # Check the depth
    depth_matches = math.isclose(1-res1.depth, 1-res2.depth,
                                 rel_tol=depthtol)
    # Check the duration
    duration_matches = math.isclose(res1.duration, res2.duration,
                                    rel_tol=durtol)

    if check_T0:
        T0_diff = abs(res1.T0 - res2.T0)
        P_ratio = T0_diff / res1.period
        difference = abs(round(P_ratio) - P_ratio)
        T0_matches = difference < P_delta / (res1.period) * P_ratio and P_ratio > 1
    else:
        T0_matches = True

    # Make sure everything matches
    return period_matches and depth_matches and duration_matches and T0_matches


def correlate_results(results_list, ptol=0.01, durtol=0.3, 
                      depthtol=0.4, check_T0=True):
    correlated_results = []
    
    # First go through and collect results based on common period
    while len(results_list) > 0:
        c_list = []
        i, j = 0, 0
        
        # Index through lists
        while i < len(results_list):
            if j == len(results_list[i]):
                i, j = i+1, 0
                continue
                
            # Check if clist is empty, if not match period of first result
            if len(c_list) == 0 or\
            math.isclose(results_list[i][j].period, c_list[0].period, rel_tol=ptol):
                c_list.append(results_list[i].pop(j))
            else:
                j += 1
        
        correlated_results.append(c_list)
        
        # Trim results:
        i = 0
        while i < len(results_list):
            if len(results_list[i]) == 0:
                results_list.pop(i)
            else:
                i += 1
    
    # Then collect sets of results with common depths and durations p harmonics
    i = 0
    while i < len(correlated_results)-1:
        p, dep, dur = get_p_dep_dur(correlated_results[i])
        j = i+1
        while j < len(correlated_results):
            p2, dep2, dur2 = get_p_dep_dur(correlated_results[j])
            p_ratio = max([p/p2, p2/p])
            lflag = len(correlated_results[i]) + len(correlated_results[j]) <=\
                    len(results_list)
            
            if math.isclose(p_ratio, round(p_ratio), rel_tol=ptol*3) and\
               math.isclose(dep, dep2, rel_tol=depthtol) and\
               math.isclose(dur, dur2, rel_tol=durtol) and lflag: 
                correlated_results[i].extend(correlated_results.pop(j))
            else:
                j += 1
        i += 1
        
    return correlated_results


def get_p_dep_dur(results_list):
    return np.mean([r.period for r in results_list]),\
           np.mean([1-r.depth for r in results_list]),\
           np.mean([r.duration for r in results_list])


def get_P_delta(result):
    P_delta = result.periods[np.argmax(result.power)+1]-\
              result.periods[np.argmax(result.power)-1]
    return P_delta


def get_result_snr(result, lc):
    """Return the SNR of a transit result from the lightcurve it was derived
    from
    """
    fnorm_binned = misc.bin_curve(lc.bjd, lc.fnorm_detrend, 
                                  lc.efnorm, even_bins=True, 
                                  bin_length=result.duration)[1]
    # Take the std
    lc_noise = np.std(fnorm_binned)
    depth = 1 - result["depth"]

    N = misc.get_Ntransits(result.period, result.T0, result.duration, lc.bjd,
                           countdiff=True)
    
    # is transit count true
    snr = depth / lc_noise * N**0.5
    
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


### Planet Candidate Vetting ###

def pc_overlap(pcs, bjd, return_cut=True):
    """Given a list of planet candidates, return a list where planet candidates
    with overlapping transits are removed. Higher SNR Planet candidates are kept 
    over lower ones. 
    """
    i = 0
    cut_pcs = []
    
    # Sort from highest to lowest based on SNR
    sort_indeces = np.argsort([pc.snr for pc in pcs])
    pcs = [pcs[i] for i in sort_indeces[::-1]]
    
    while i < len(pcs)-1:
        j = i+1
        intransit = transit_mask(bjd, pcs[i].period, 
                                 pcs[i].duration, pcs[i].T0)
        while j < len(pcs):
            intransit2 = transit_mask(bjd, pcs[j].period, 
                                      pcs[j].duration, pcs[j].T0)

            # 100% of duration in ~2 minutes to check overlap
            # Get longest intransit part
            max_overlap = longest_overlap(intransit, intransit2)
            
            if max_overlap*2/(24*60) > 0.9*pcs[i].duration: 
                cut_pcs.append(pcs.pop(j))
            else:
                j += 1
        
        i += 1
    
    if return_cut:
        return pcs, cut_pcs
    
    return pcs


def longest_overlap(intransit1, intransit2):
    """Return the length of the longest overlap in the two boolean arrays
    """
    overlap = intransit1 & intransit2
    
    overlap_length = 0
    max_overlap = 0
    
    for i in range(len(overlap)):
        if overlap[i]:
            overlap_length += 1
        else:
            max_overlap = max(max_overlap, overlap_length)
            overlap_length = 0
    
    return max_overlap
