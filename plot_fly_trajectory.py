# (c) 2013 Floris van Breugel
#
# This script is programmed to read a file, "sim_data.pickle," created by the simulation "fly_plume_sim.py," and make the corresponding figure/animation. 
# The code relies on plotting packages that are freely available from https://github.com/florisvb

import pickle
import fly_plot_lib.plot as fpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def get_frame_name(frame, nframes):
    ndigits = len(str(nframes)) + 1
    name = str(frame)
    while len(name) < ndigits:
        name = '0' + name
    return name
    

if __name__ == '__main__':
    # options
    figure_format = 'pdf' # alternative: png
    axes=[0,1] # alternative: [0,1]
    
    f = open('sim_data.pickle')
    data = pickle.load(f)
    f.close()
    
    nframes = len(data['fly'])
    folder = 'frames/'
    frame_where_fly_appears = 0
    for i in range(nframes):
        if len(data['fly'][i]):
            frame_where_fly_appears = i
            break
    print 'frame of fly appearance: ', frame_where_fly_appears
    
    frames = np.arange(0,nframes,50)
    frames = np.hstack((frames, nframes-1))
    for frame in frames:
        fig = plt.figure(figsize=(3.3,1.5))
        ax = fig.add_axes([0,0,1,1])
        ax.set_frame_on(False)
        
        # plot odor packets
        for odor_packet in data['odor_packets'][frame]:
            x = odor_packet[axes[0]]
            y = odor_packet[axes[1]]
            odor_packet = patches.Circle((x,y), odor_packet[-1], edgecolor='white', facecolor='red', alpha=0.3)
            ax.add_artist(odor_packet)
            
        # plot fly trail
        if frame > frame_where_fly_appears:
            xpos = []
            ypos = []
            
            for i in range(frame_where_fly_appears, frame):
                fly = data['fly'][i]
                if len(fly) > 0:
                    xpos.append( fly[axes[0]] )
                    ypos.append( fly[axes[1]] )
            ax.plot(xpos, ypos, color='black')
            
        # plot visual feature
        x = data['visual_features'][0][axes[0]]
        y = data['visual_features'][0][axes[1]]
        visual_feature = patches.Circle((x,y), 10, color='black')
        ax.add_artist(visual_feature)
            
        # make pretty
        ax.set_xlim(-1200, 200)
        ax.set_ylim(-400, 400)
        ax.set_aspect('equal')
        
        ax.hlines(-300, -1000, 0, color='black')
        ax.text(-1000, -295, '1 m', verticalalignment='bottom', horizontalalignment='left')
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        figname = folder + get_frame_name(frame, nframes) + figure_format
        if figure_format == 'pdf':
            fig.savefig(figname, format='pdf')
        elif figure_format == 'png':
            fig.savefig(figname, format='png')
        
        print frame

    if figure_format == 'png':
        print 'To turn the PNGs into a movie, you can run this command from inside the directory with the tmp files: '
        print 'mencoder \'mf://*.png\' -mf type=png:fps=30 -ovc lavc -lavcopts vcodec=mpeg4 -oac copy -o animation.avi'
