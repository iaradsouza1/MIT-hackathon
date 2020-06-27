import os
import sys
import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
from matplotlib.dates import date2num, num2date
from matplotlib import dates as mdates
from matplotlib import ticker
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from datetime import date

from scipy import stats as sps
from scipy.interpolate import interp1d

import downloaddata as apid

# SET UP PARAMETERS
k = np.array([20, 40, 55, 90])

# We create an array for every possible value of Rt
R_T_MAX = 12
r_t_range = np.linspace(0, R_T_MAX, R_T_MAX*100+1)

# Gamma is 1/serial interval
# https://wwwnc.cdc.gov/eid/article/26/6/20-0357_article
GAMMA = 1/4

states_names = [
    'AC', 'AL', 'AM', 'AP',
    'BA', 'CE', 'DF', 'ES',
    'GO', 'MA', 'MG', 'MS',
    'MT', 'PA', 'PB', 'PE',
    'PI', 'PR', 'RJ', 'RN',
    'RO', 'RR', 'RS', 'SC',
    'SE', 'SP', 'TO'
]

no_lockdown = [
    'AC', 'GO', 'MS', 'MT',
    'PI', 'PR', 'RR', 'RS',
    'SC', 'SP'
]

partial_lockdown = [
    'AL', 'AM', 'AP', 'BA',
    'CE', 'DF', 'ES', 'MA',
    'MG', 'PA', 'PB', 'PE',
    'RJ', 'RN', 'RO', 'SE',
    'TO'
]

FILTERED_REGIONS = []

FULL_COLOR = [.7,.7,.7]
NONE_COLOR = [179/255,35/255,14/255]
PARTIAL_COLOR = [.5,.5,.5]
ERROR_BAR_COLOR = [.3,.3,.3]

def fetch_data(file_name):
    
    if os.path.isfile(file_name):
        print ("File exist")
    else:
        print ("Dowloading Data")
        apid.main()
        print(f"Done, today {date.today()} Data has been downloaded")


def load_data(file_name):
    raw_states_df = pd.read_csv(file_name,
                     usecols=[1,3,4,7],
                     parse_dates=['date'],
                     squeeze=True
                     ).sort_index()

    raw_states_df['cases']= raw_states_df['last_available_confirmed']
    raw_states_df.drop(['last_available_confirmed'],axis=1, inplace=True)
    raw_states_df.dropna(inplace=True)
    raw_states_df.drop(['city'],axis=1, inplace=True)

    if not os.path.isdir("./metrics"):
        os.mkdir("./metrics")
    
    raw_states_df.to_pickle("metrics/raw_states_df.pkl")
    return raw_states_df


def squeeze_df(raw_states_df):
    """ Group Df by state date and convert to pd.Series"""
    states_df = raw_states_df.groupby(['state', 'date']).sum()
    states = states_df.squeeze()

    states.to_pickle("metrics/state.pkl")
    return states


def prepare_cases(cases, state_name):
    new_cases = cases.diff()

    smoothed = new_cases.rolling(7,
        win_type='gaussian',
        min_periods=1,
        center=True).mean(std=2).round()
    
    zeros = smoothed.index[smoothed.eq(0)]
    if len(zeros) == 0:
        idx_start = 0
    else:
        last_zero = zeros.max()
        idx_start = smoothed.index.get_loc(last_zero) + 1
    smoothed = smoothed.iloc[idx_start:]
    original = new_cases.loc[smoothed.index]

    # Pickle the two series
    original.to_pickle("metrics/" + state_name + "_original.pkl")
    smoothed.to_pickle("metrics/" + state_name + "_smoothed.pkl")

    return original, smoothed


def get_posteriors(sr, state_name,  window=7, min_periods=1) :
    lam = sr[:-1].values * np.exp(GAMMA * (r_t_range[:, None] - 1))

    # Note: if you want to have a Uniform prior you can use the following line instead.
    # I chose the gamma distribution because of our prior knowledge of the likely value
    # of R_t.
    
    # prior0 = np.full(len(r_t_range), np.log(1/len(r_t_range)))
    prior0 = np.log(sps.gamma(a=3).pdf(r_t_range) + 1e-14)

    likelihoods = pd.DataFrame(
        # Short-hand way of concatenating the prior and likelihoods
        data = np.c_[prior0, sps.poisson.logpmf(sr[1:].values, lam)],
        index = r_t_range,
        columns = sr.index)

    # Perform a rolling sum of log likelihoods. This is the equivalent
    # of multiplying the original distributions. Exponentiate to move
    # out of log.
    posteriors = likelihoods.rolling(window,
                                     axis=1,
                                     min_periods=min_periods).sum()
    posteriors = np.exp(posteriors)

    # Normalize to 1.0
    posteriors = posteriors.div(posteriors.sum(axis=0), axis=1)
    
    posteriors.to_pickle("metrics/" + state_name + "_posteriors.pkl")
    return posteriors


def highest_density_interval(pmf, state_name, p=.95 ):
    low = ''
    # If we pass a DataFrame, just call this recursively on the columns
    if(isinstance(pmf, pd.DataFrame)):
        return pd.DataFrame([highest_density_interval(pmf[col],state_name) for col in pmf],
                            index=pmf.columns)
    
    cumsum = np.cumsum(pmf.values)
    best = None
    for i, value in enumerate(cumsum):
        for j, high_value in enumerate(cumsum[i+1:]):
            if (high_value-value > p) and (not best or j<best[1]-best[0]):
                best = (i, i+j+1)
                break
    
    # I had to do this because some points in best are None, due to the data quality
    if best is not None:     
        low = pmf.index[best[0]]
        high = pmf.index[best[1]]
    else:
        low = float('nan')
        high = float('nan')

    hdis= pd.Series([low, high], index=['Low', 'High']).to_pickle("metrics/" + state_name + "_hdi.pkl")

    return pd.Series([low, high], index=['Low', 'High'])


def plot_rt(result, ax,fig, state_name):
    
    ax.set_title(f"{state_name}")
    
    # Colors
    ABOVE = [1,0,0]
    MIDDLE = [1,1,1]
    BELOW = [0,0,0]
    cmap = ListedColormap(np.r_[
        np.linspace(BELOW,MIDDLE,25),
        np.linspace(MIDDLE,ABOVE,25)
    ])
    color_mapped = lambda y: np.clip(y, .5, 1.5)-.5
    
    index = result['ML'].index.get_level_values('date')
    values = result['ML'].values
    
    # Plot dots and line
    ax.plot(index, values, c='k', zorder=1, alpha=.25)
    ax.scatter(index,
               values,
               s=40,
               lw=.5,
               c=cmap(color_mapped(values)),
               edgecolors='k', zorder=2)
    
    # Aesthetically, extrapolate credible interval by 1 day either side
    lowfn = interp1d(date2num(index),
                     result['Low'].values,
                     bounds_error=False,
                     fill_value='extrapolate')
    
    highfn = interp1d(date2num(index),
                      result['High'].values,
                      bounds_error=False,
                      fill_value='extrapolate')
    
    extended = pd.date_range(start=pd.Timestamp('2020-03-01'),
                             end=index[-1]+pd.Timedelta(days=1))
    
    ax.fill_between(extended,
                    lowfn(date2num(extended)),
                    highfn(date2num(extended)),
                    color='k',
                    alpha=.1,
                    lw=0,
                    zorder=3)

    ax.axhline(1.0, c='k', lw=1, label='$R_t=1.0$', alpha=.25);
    
    # Formatting
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
    ax.yaxis.tick_right()
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.margins(0)
    ax.grid(which='major', axis='y', c='k', alpha=.1, zorder=-2)
    ax.margins(0)
    ax.set_ylim(0.0,3.5)
    ax.set_xlim(pd.Timestamp('2020-03-01'), result.index.get_level_values('date')[-1]+pd.Timedelta(days=1))
    fig.set_facecolor('w')
    

def analyze_all(states):

    results = {}

    states_to_process = states.loc[~states.index.get_level_values('state').isin(FILTERED_REGIONS)]

    for state_name, cases in states_to_process.groupby(level='state'):

        print(f'Processing {state_name}')
        original, smoothed = prepare_cases(cases,state_name)
        
        print("Plotting New Cases per Day")
        plot_smoothed(original, smoothed, state_name)

        print('\tGetting Posteriors')
        try:
            posteriors = get_posteriors(smoothed,state_name)
            print("Plotting posteriors")
            plot_posteriors(posteriors, state_name)
        except:
            display(cases)
        print('\tGetting HDIs')
        hdis = highest_density_interval(posteriors,state_name)
      
            
        print('\tGetting most likely values')
        most_likely = posteriors.idxmax().rename('ML')
        result = pd.concat([most_likely, hdis], axis=1)
        results[state_name] = result.droplevel(0)

    print("Plotting hdis")
    for i, (state_name, result) in enumerate(results.items()):
        plot_hdis(result, state_name)

    # Plot all states
    print("Plotting All states together")
    _plot_all_states(results)

    # Pickle restults
    # Use this to unpickle
    ##################################################
        #with open('results.pkl', 'rb') as handle:
            #results = pickle.load(handle)
    ###################################################
    with open('results.pkl', 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return results


def plot_smoothed(original, smoothed, state_name):
        
    original.plot(title=f"{state_name} New Cases per Day",
               c='k',
               linestyle=':',
               alpha=.5,
               label='Actual',
               legend=True,
             figsize=(600/72, 400/72))

    if not os.path.isdir("./plot"):
        os.mkdir("./plot")

    ax = smoothed.plot(label='Smoothed',
                   legend=True)
    ax.get_figure().set_facecolor('w')
    plt.savefig("plot/" + state_name + "_new_cases_per_day.png")
    plt.close()


def plot_posteriors(posteriors, state_name):

    ax = posteriors.plot(title=f'{state_name} - Daily Posterior for $R_t$',
           legend=False, 
           lw=1,
           c='k',
           alpha=.3,
           xlim=(0.4,4))

    ax.set_xlabel('$R_t$')
    plt.savefig("plot/" + state_name + "_posteriors.png")
    plt.close()


def plot_hdis(result, state_name):
  
    fig, ax = plt.subplots(figsize=(600/72,400/72))

    plot_rt(result, ax, fig, state_name)
    ax.set_title(f'Real-time $R_t$ for {state_name}')
    ax.set_ylim(.5,3.5)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.savefig("plot/" + state_name + "_hdis.png")
    plt.close()


def _plot_all_states(results):
    ncols = 4
    nrows = int(np.ceil(len(results) / ncols))

    # fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*3))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows*3))

    for i, (state_name, result) in enumerate(results.items()):
        plot_rt(result,  axes.flat[i], fig, state_name)

    fig.tight_layout()
    fig.set_facecolor('w')
    plt.savefig("plot/All.png")
    plt.close()


def compute_overall(results):
    overall = None

    for state_name, result in results.items():
        r = result.copy()
        r.index = pd.MultiIndex.from_product([[state_name], result.index])
        if overall is None:
            overall = r
        else:
            overall = pd.concat([overall, r])

    overall.sort_index(inplace=True)

    # Pickle overall
    overall.to_pickle("metrics/overall.pkl")

    filtered = overall.index.get_level_values(0).isin(FILTERED_REGIONS)
    np.save("metrics/filtered", filtered)  # must be loaded with np.load()
    

    mr = overall.loc[~filtered].groupby(level=0)[['ML', 'High', 'Low']].last()
    mr.to_pickle("metrics/mr.pkl")

    return mr


def plot_mr(mr):

    mr.sort_values('ML', inplace=True)
    fig, ax = plot_standings(mr)
    plt.savefig("plot/ML.png")
    plt.close
    
    mr.sort_values('High', inplace=True)
    plot_standings(mr)
    plt.savefig("plot/High.png")
    plt.close

    show = mr[mr.High.le(1.1)].sort_values('ML')
    fig, ax = plot_standings(show, title='Likely Under Control')
    plt.savefig("plot/undercontrol.png")
    plt.close

    show = mr[mr.Low.ge(1.05)].sort_values('Low')
    fig, ax = plot_standings(show, title='Likely Not Under Control');
    ax.get_legend().remove()    
    plt.savefig("plot/not_undercontrol.png")
    plt.close


def plot_standings(mr, figsize=None, title='Most Recent $R_t$ by State'):
    if not figsize:
        figsize = ((15.9/50)*len(mr)+.1,2.5)
        
    fig, ax = plt.subplots(figsize=figsize)

    ax.set_title(title)
    err = mr[['Low', 'High']].sub(mr['ML'], axis=0).abs()
    bars = ax.bar(mr.index,
                  mr['ML'],
                  width=.825,
                  color=FULL_COLOR,
                  ecolor=ERROR_BAR_COLOR,
                  capsize=2,
                  error_kw={'alpha':.5, 'lw':1},
                  yerr=err.values.T)

    for bar, state_name in zip(bars, mr.index):
        if state_name in no_lockdown:
            bar.set_color(NONE_COLOR)
        if state_name in partial_lockdown:
            bar.set_color(PARTIAL_COLOR)

    labels = mr.index.to_series().replace({'District of Columbia':'DC'})
    ax.set_xticklabels(labels, rotation=90, fontsize=11)
    ax.margins(0)
    ax.set_ylim(0,2.)
    ax.axhline(1.0, linestyle=':', color='k', lw=1)

    leg = ax.legend(handles=[
                        Patch(label='Full', color=FULL_COLOR),
                        Patch(label='Partial', color=PARTIAL_COLOR),
                        Patch(label='None', color=NONE_COLOR)
                    ],
                    title='Lockdown',
                    ncol=3,
                    loc='upper left',
                    columnspacing=.75,
                    handletextpad=.5,
                    handlelength=1)

    leg._legend_box.align = "left"
    fig.set_facecolor('w')
    return fig, ax


def main():

    # Acquire data
    # Save as cvs
    today = date.today()
    file_name = "Covid-19-Brasil_" + str(today) + ".csv"
    if os.path.isfile(file_name):
        print ("File exist")
    else:
        apid.call_request()

    #Load and parse csv and
    raw_states_df = load_data("data/Covid-19-Brasil_Current.csv") 

    # Convert to Series 
    states = squeeze_df(raw_states_df)
    
    # Run analysis to all states
    results = analyze_all(states)

    mr = compute_overall(results)
    plot_mr(mr)

if __name__ == "__main__":
    main()