import numpy as np
import matplotlib.pyplot as plt

import batman
import emcee
import scipy
import corner

def transit_log_prob(params, star_params, lc_arrays, param_priors):
    
    # Get probability of model
    prior_probs = [param_priors[i].pdf(params[i]) for i in range(4)]
    if 0 in prior_probs:
        # print(prior_probs)
        return -np.inf
    
    prior_prob = np.sum(np.log(np.array(prior_probs)))
    
    # Parameter Order
    T0, P, Rp, b = params
    bjd, fnorm, efnorm = lc_arrays
    R, R_err, M, M_err, u = star_params
    
    # Fold Lightcurve, make sure this is working
    bjd_folded = (bjd - T0 + P/2) % P - P/2
    
    # Could enforce all points at zero here
    
    # Sample a in units of R* Semimajor axis / R*
    R_star = np.random.normal(R, R_err)
    M_star = np.random.normal(M, M_err)
    
    # G in R⊙^3 / M⊙ days^2
    # Should make this a function
    G = 2942.2062
    a = (P**2 * M_star / (4*np.pi**2) * G)**(1/3) / R_star
    # Compute inclination from 
    inc = np.arccos(b / a) * 180/np.pi
    
    # Initialize Batman Transit
    bm_params = batman.TransitParams()
    bm_params.per, bm_params.rp, bm_params.inc = P, Rp, inc
    bm_params.t0 = 0
    bm_params.limb_dark = "quadratic"
    bm_params.u = u
    bm_params.a = a
    bm_params.ecc = 0
    bm_params.w = 90
    
    # Make Model
    m = batman.TransitModel(bm_params, bjd_folded)
    
    # Compute Log Posterior probability
    chi2 = -0.5 * sum((m.light_curve(bm_params) -  fnorm)**2 / efnorm**2)
    
    # Good up to an additive constant
    # print(chi2, prior_prob)
    
    return chi2 + prior_prob


def get_priors_data(ts, correlated_results):
    if len(correlated_results) > 1:
        T0s = np.array([result['T0'] for result in correlated_results])
        Ps = np.array([result['period'] for result in correlated_results])
        # Find all differences in T0
        T0_diff_array = np.array([np.abs(T0s - T0) for T0 in T0s])
        
        # Take all Ps that are close to the max P
        P_guess = np.mean(Ps[np.isclose(Ps, max(Ps), rtol=0.1)])
        
        # This could be faulty, what if the P_guess is bad?
        ratio_array = np.round(T0_diff_array / P_guess)
        
        # The ratios should be close to integers
        P_guess_array = np.divide(T0_diff_array, ratio_array, where=(ratio_array!=0))
        P_guess_array = P_guess_array[P_guess_array > 0]
        
        T0 = min(T0s)
        P = np.mean(P_guess_array)
        # P_unround = np.divide(T0_diff_array, T0_diff_array/P_guess,
        #                       where=(T0_diff_array!=0))
        # P_sigma = abs(np.mean(P_unround[P_unround > 0]) - P)
        # P_guess_unround = np.divide(T0_diff_array, T0_diff_array / P_guess, 
        #                             where=(T0_diff_array!=0))
        # P_unround
        P_sigma = abs(P - P_guess)
        
    else:
        result = correlated_results[0]
        P = result['period']
        T0 = result['T0']
        P_sigma = None
    
    # Get star info, set duration guess
    R, R_err, M, M_err = ts.radius, ts.radius_err, ts.mass, ts.mass_err
    G = 2942.2062
    a = (P**2 * M / (4*np.pi**2) * G)**(1/3) / R
    duration = P / np.pi * np.arcsin(1/a)
    
    # Set sigmas, don't 
    T0_sigma = duration
    if P_sigma is None:
        P_sigma = duration
    
    # Get lightcurves
    bjd, fnorm, efnorm = np.concatenate([lc.bjd for lc in ts.lightcurves]),\
                np.concatenate([lc.fnorm_detrend for lc in ts.lightcurves]),\
                np.concatenate([lc.efnorm for lc in ts.lightcurves])
    
    # Mask everything 10 times from the duration, should revisit this
    bjd_folded = (bjd - T0 + P/2) % P - P/2
    cut = np.abs(bjd_folded) < 10*duration
    bjd_c, fnorm_c, efnorm_c = bjd[cut], fnorm[cut], efnorm[cut]
    
    # Setting priors
    # T0 fraction of transit duration 
    T0_model = scipy.stats.norm(T0, T0_sigma)
    # Period prior informed by period grid? It already has uncertainty
    P_model = scipy.stats.norm(P, P_sigma)
    # Uniform in log space? Linear log cutoff depth sets max Rp/R*
    Rp_model = scipy.stats.loguniform(0.005, 0.5)
    # Uniform prior on impact parameter?
    b_model = scipy.stats.uniform(0, 0.95)
    
    priors = (T0_model, P_model, Rp_model, b_model)
    star_params = (ts.radius, ts.radius_err, ts.mass, ts.mass_err, ts.u)
    lc_arrays = (bjd_c, fnorm_c, efnorm_c)
    
    return priors, star_params, lc_arrays


def get_pos(nwalkers, correlated_results, priors):
    # Get depth estimates from results
    depths = [(1-result.depth)**0.5 for result in correlated_results]
    Rp, Rp_sigma = np.mean(depths), np.std(depths)
    
    # Get samplers to sample walker positions from
    Rp_dist = scipy.stats.norm(Rp, Rp_sigma)
    T0_dist, P_dist, b_dist = priors[0], priors[1], priors[3]
    walker_samplers = [T0_dist, P_dist, Rp_dist, b_dist]
    
    # Sample Walker positions
    # walker_pos = [sampler.stats(moments='m') for sampler in walker_samplers]
    # walkers = np.array(walker_pos*nwalkers).reshape(nwalkers, 4)
    # walkers *= np.random.uniform(0.99, 1.01, size=walkers.shape)
    
    walker_pos = [sampler.rvs(size=nwalkers) for sampler in walker_samplers]
    walkers = np.array(walker_pos).T
        
    return walkers


def mcmc_planet_fit(ts, correlated_results, steps=5000, nwalkers=48):
    priors, star_params, lc_arrays = get_priors_data(ts, correlated_results)
    walkers = get_pos(nwalkers, correlated_results, priors)
    
    ensam = emcee.EnsembleSampler(nwalkers, 4, transit_log_prob, 
                                  args=(star_params, lc_arrays, priors))
    
    ensam.run_mcmc(walkers, nsteps=steps, progress=True)
    return ensam, priors, lc_arrays, star_params


### Plotting Functions ###

def plot_chain_dists(chain, priors, titles, savefig=None, square=True, T0_offset=True):
    if T0_offset:
        T0_m = int(np.mean(chain[:,0]))
        offset = np.zeros(chain.shape)
        offset[:,0] += np.ones(len(offset[:,0])) * T0_m
        chain = chain - offset
        titles = titles.copy()
        titles[0] += f"+{T0_m}"
    else:
        T0_m = 0

    fig, axs = plt.subplots(1, len(priors), figsize=(4*len(priors), 4))
    fig.subplots_adjust(wspace=0.03)
    
    for i in range(len(axs)):
        ext = max(chain[:,i]) - min(chain[:,i])
        x = np.linspace(min(chain[:,i])-ext/10, 
                        ext/10+max(chain[:,i]), num=200)
        if i == 0:
            pdf = priors[i].pdf(x + T0_m)
        else:
            pdf = priors[i].pdf(x)
        
        n,_,_=axs[i].hist(chain[:,i], density=True, bins=100)
        axs[i].plot(x,max(n)/max(pdf)*pdf/2, ls='--', alpha=0.7)
        axs[i].set_title(titles[i])
        axs[i].set_xticks(axs[i].get_xticks(), axs[i].get_xticklabels(), 
                          rotation=45, ha='right')              
        axs[i].set_xlim(min(x), max(x))
        axs[i].set_yticks([])
    
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
    plt.show()
    
    
    
def plot_chain_corner(chain, labels, savefig=None, T0_offset=True):
    if T0_offset:
        T0_m = int(np.mean(chain[:,0]))        
        offset = np.zeros(chain.shape)
        offset[:,0] += np.ones(len(offset[:,0])) * T0_m
        chain = chain - offset
        labels = labels.copy()
        labels[0] += f"+{T0_m}"
        
    figure = corner.corner(chain, labels=labels)
    
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
        
    plt.show()
    


def plot_models(lc_arrays, star_params, chain, samples=20, savefig=None):
    plt.figure(figsize=(12,4))
    
    bjd, fnorm, efnorm = lc_arrays
    R, R_err, M, M_err, u = star_params
    T0, P, Rp, b = np.median(chain, axis=0)
    bjd_folded = (bjd - T0 + P/2) % P - P/2
    plt.scatter(bjd_folded, fnorm, s=1)
    
    sort = np.argsort(bjd_folded)
    bjd_folded, fnorm = bjd_folded[sort], fnorm[sort]
    
    bins = np.arange(0, len(bjd_folded), 50)
    fnorm_mean, fnorm_std = np.array([[np.mean(fnorm[bins[i]:bins[i+1]]), 
                                       np.std(fnorm[bins[i]:bins[i+1]])] 
                                       for i in range(len(bins)-1)]).T
    bjd_mean, bjd_std = np.array([[np.mean(bjd_folded[bins[i]:bins[i+1]]), 
                                   np.std(bjd_folded[bins[i]:bins[i+1]])] 
                                  for i in range(len(bins)-1)]).T
    plt.errorbar(bjd_mean, fnorm_mean, fnorm_std/50**0.5, ls='', 
                 capsize=3, marker='.', color='red')
    
    fold_sort = np.argsort(bjd_folded)
    for k in np.random.choice(np.arange(len(chain)), samples):
        # Parameter Order
        T0_s, P, Rp, b = chain[k]
        
        # G in R⊙^3 / M⊙ days^2
        # Should make this a function
        G = 2942.2062
        a = (P**2 * M / (4*np.pi**2) * G)**(1/3) / R
        # Compute inclination from 
        inc = np.arccos(b / a) * 180/np.pi

        # Initialize Batman Transit
        bm_params = batman.TransitParams()
        bm_params.per, bm_params.rp, bm_params.inc = P, Rp, inc
        bm_params.t0 = T0 - T0_s
        bm_params.limb_dark = "quadratic"
        bm_params.u = u
        bm_params.a = a
        bm_params.ecc = 0
        bm_params.w = 90

        # Make Model
        m = batman.TransitModel(bm_params, bjd_folded)
        
        plt.plot(bjd_folded[fold_sort], m.light_curve(bm_params)[fold_sort], 
                 color='k', lw=1)
    
    duration = P / np.pi * np.arcsin(1/a)
    plt.xlim(-1.5*duration, 1.5*duration)
    plt.ylabel("Normalized Flux")
    plt.xlabel("Days Since Transit Middle")
    
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
        
    plt.show()
    

def plot_chain_evo(chain, titles, savefig=None):
    # Chain must be staight from ensam
    fig, axs = plt.subplots(1, chain.shape[-1], 
                            figsize=(4*chain.shape[-1], 4), sharey=False)
    fig.subplots_adjust(wspace=0.17)
    for i in range(len(axs)):
        for k in range(chain.shape[1]):
            axs[i].plot(np.arange(len(chain[:,k,i])),chain[:,k,i], 
                        alpha=0.2, lw=0.1, color='0.2')
        axs[i].plot(np.arange(len(chain[:,0,i])),
                    np.median(chain[:,:,i], axis=1), lw=1, color='r')
        axs[i].set_title(titles[i])
        axs[i].set_xlabel('Step Number')
        axs[i].set_yticks(axs[i].get_yticks(), axs[i].get_yticklabels(), 
                          rotation=70, ha='right')
        axs[i].set_xlim(0, chain.shape[0])
        
    axs[0].set_ylabel('Parameter Value')
    # plt.tight_layout()
    
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')
        
    plt.show()
    
    
def confidence_interval(chain, perc=0.95):
    """Compute the confidence interval"""
    intervals = []
    b = (1-perc)/2
    
    for dat in chain.T:
        sorted_dat = np.sort(dat)
        index = int(len(sorted_dat)*b)
        
        left = sorted_dat[index]
        right = sorted_dat[-max(1, index)]
        
        intervals.append([left, right])
        
    return intervals    
