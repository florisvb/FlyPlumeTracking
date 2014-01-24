This simulation was written for the following paper describing the odor plume tracking behavior of fruit flies: van Breugel, F. & Dickinson, M. H. (2014) Plume tracking behavior of flying Drosophila emerges from a set of distinct sensory-motor reflexes. Current Biology: http://www.sciencedirect.com/science/article/pii/S0960982213015820


The code depends on several other packages that are either freely available on the internet, or freely available within this git repository: https://github.com/florisvb

Author(s):

    * Floris van Breugel (florisvb at gmail dot com)

This package may be freely distributed and modified in accordance with the GNU General Public License v3.


Running the simulation:

Open fly_plume_sim.py, and scroll down to the main script, labelled by the following line:

if __name__ == '__main__':

The save_data variable toggles saving the plume packet data, to make animations of the simulation (True), or to run the simulation more quickly to test large amounts of iterations (False).

If save_data is False, the script will run through three simulations with different parameters and save the results to "sim_results.pickle." 

After running the simulation, run "plot_simulation_results.py" to open this file, and plot the results. 

