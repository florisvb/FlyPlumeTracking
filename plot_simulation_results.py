# (c) 2013 Floris van Breugel
#
# This script is programmed to read a file, "sim_results.pickle," created by the simulation "fly_plume_sim.py," and make the corresponding figure. 
# The code relies on plotting packages that are freely available from https://github.com/florisvb

# Set plotting parameters
import fly_plot_lib
params = {   'font.size' : 8,
             'text.fontsize': 8,
             'axes.labelsize': 8,
             'xtick.labelsize': 8,
             'ytick.labelsize': 8,
             }
fly_plot_lib.set_params.pdf(params)

import pickle
import fly_plot_lib.plot as fpl
import fly_plot_lib.text as flytext
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    
    f = open('sim_results.pickle')
    results = pickle.load(f)
    f.close()
    
    fig = plt.figure(figsize=(3.3,2))
    ax = fig.add_axes([0.05,0.25,0.9,0.65])
    
    print len(results)
    
    colors = ['black', 'orange', 'purple']
    
    fpl.histogram(ax, results, colors=colors[0:len(results)], bins=20, normed=True, show_smoothed=False, bar_alpha=1, bin_width_ratio=1)
    
    xticks = np.linspace(0,20000,11)
    xticklabels = [str(int(tick/1000.)) for tick in xticks]
    fpl.adjust_spines(ax, ['bottom'], xticks=xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('Time to localization, sec')
    
    flytext.set_fontsize(fig, 8)
    
    fig.savefig('results_plot.pdf', format='pdf')
