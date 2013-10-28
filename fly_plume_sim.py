"""
-----------------------------------------------------------------------
fly_plume_sim
Copyright (C) Floris van Breugel, 2013.
  
florisvb@gmail.com

Released under the GNU GPL license, Version 3


fly_plume_sim is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
    
fly_plume_sim is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
License for more details.

You should have received a copy of the GNU General Public
License along with fly_plume_sim.  If not, see <http://www.gnu.org/licenses/>.

------------------------------------------------------------------------
"""

########################################################################
# Notes:
#
# Units: mm, ms
########################################################################

import numpy as np
import copy
import pickle

### Helper functions ###################################################
def get_heading(velocity, wind):
    a = np.arctan2(velocity[0], velocity[1]) - np.arctan2(wind[0], wind[1])
    while a < -1*np.pi:
        a += np.pi
    while a > np.pi:
        a -= np.pi
    return a

def rotation_matrix(axis,theta):
    axis = axis/np.linalg.norm(axis)
    a = np.cos(theta/2.)
    b,c,d = (-axis*np.sin(theta/2.)).tolist()[0]
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

def dist_point_to_line(pt, linept1, linept2, resolution=10):
    
    xs = np.linspace(linept1[0], linept2[0], resolution, endpoint=True)    
    ys = np.linspace(linept1[1], linept2[1], resolution, endpoint=True)    
    zs = np.linspace(linept1[2], linept2[2], resolution, endpoint=True)    
    
    pts = np.vstack((xs,ys,zs))
    
    distances = []
    for i in range(resolution):
        d = np.linalg.norm(pt-pts[:,i].reshape(3,1))
        distances.append(d)
    return np.min(distances)
########################################################################



### Simulation Classes #################################################
class VisualFeature(object):
    def __init__(self, position, radius):
        self.position = position
        self.radius = radius
        
class OdorPacket(object):
    def __init__(self, position, radius):
        self.position = position
        self.radius = radius
    def update(self, wind_vector, dt):
        self.position = self.position + wind_vector*dt
    def is_point_inside(self, point):
        v = np.linalg.norm(self.position - point)
        if v < self.radius:
            return True
        else:
            return False
            
class Fly(object):
    def __init__(self, position, fly_parameters=None):
        # initialize
        self.position = position
        self.time = 0
        self.time_at_last_encounter = -1
        self.altitude_sign = 1
        self.casting_direction = 1
        self.casting_initiated = False
        self.surge_initiated = False
        self.time_climb_reversed = 10000
        self.time_cast_initiated = 10000
        self.time_entered_plume = 10000
        self.time_exited_plume = 0
        self.odor = False
        self.ground_velocity_vector = np.zeros([3,1])
        
        self.history_position = []
        self.history_velocity = []
        self.history_heading = []
        self.history_odor = []
        self.history_wind = []
        
        # current behavior
        self.behavior = 'surge'
        
        # behavioral parameters
        self.surge_delay = fly_parameters['surge_delay']
        self.cast_delay = fly_parameters['cast_delay']
        self.cast_time_amplitude = fly_parameters['cast_time_amplitude']
        self.altitude_time_amplitude = fly_parameters['altitude_time_amplitude']
        self.visual_attraction_probability = fly_parameters['visual_attraction_probability']
        self.visual_attraction_distance = 200
        self.landing_distance = 5

    def update(self, odor, wind, dt, visual_features):
        self.time += dt
        self.wind = wind

        if odor is True: # initiate surging, with delay
            self.time_at_last_encounter = self.time
            if self.surge_initiated is False:
                self.surge_initiated = True
                self.time_entered_plume = self.time
        if odor is False: # initiate casting, with delay
            if (self.time - self.time_at_last_encounter) < 10*1000 and not self.casting_initiated:
                self.casting_initiated = True
                self.climb_reversed = False
                self.time_exited_plume = self.time
        self.odor = odor
        
        # surge
        if self.surge_initiated and self.behavior != 'surge':
            if self.time - self.time_entered_plume > self.surge_delay:
                self.casting_initiated = False
                self.altitude_initiated = False
                self.surge_initiated = False
                self.behavior = 'surge'
                
        # cast
        if self.casting_initiated and self.behavior != 'cast':
            if self.time - self.time_exited_plume > self.cast_delay:
                self.time_climb_reversed = self.time - self.altitude_time_amplitude/2. 
                self.time_cast_initiated = self.time - self.cast_time_amplitude/2.
                self.casting_initiated = False
                self.surge_initiated = False
                self.behavior = 'cast'

        # control casting reversals for horizontal and vertical directions
        if self.behavior == 'cast':
            if self.time - self.time_climb_reversed > self.altitude_time_amplitude:
                self.altitude_sign = np.sign(self.ground_velocity_vector[2])*-1
                self.time_climb_reversed = self.time

            if self.time - self.time_cast_initiated > self.cast_time_amplitude:
                self.casting_direction *= -1
                self.time_cast_initiated = self.time
                
        # calculate distance to visual feature
        distance_to_visual_features = []
        for visual_feature in visual_features:
            distance = np.linalg.norm(self.position - visual_feature.position)
            distance_to_visual_features.append(distance)
        min_distance = np.min(distance_to_visual_features)
        
        # visual attraction behavior
        if self.odor is False:
            if min_distance < self.visual_attraction_distance:
                r = np.random.random()
                if r < self.visual_attraction_probability:
                    self.behavior = 'visual_attraction'
                    visual_object_index = np.argmin(distance_to_visual_features)
                    visual_object = visual_features[visual_object_index].position
                else:
                    self.behavior = 'cast'
        elif self.behavior == 'visual_attraction':
            self.behavior = 'cast'
                
        # land if close to object
        if min_distance < self.landing_distance:
            if odor:
                self.behavior = 'land'
            else:
                self.behavior = 'cast'
            
        # set velocity vector based on behavior
        if self.behavior == 'surge':
            self.ground_velocity_vector = -2*self.wind
        if self.behavior == 'cast':
            altitude_velocity = self.altitude_sign*0.4
            # upwind, adjust altitude velocity
            self.ground_velocity_vector = -1*wind
            self.ground_velocity_vector[2] = altitude_velocity
            # now make crosswind
            x,y = copy.copy(self.ground_velocity_vector[0:2])
            self.ground_velocity_vector[0] = y*self.casting_direction*-1
            self.ground_velocity_vector[1] = x*self.casting_direction
            # add small upwind component to make plots more interpretable
            self.ground_velocity_vector += -0.04*self.wind 
        if self.behavior == 'visual_attraction':
            vector_to_object = visual_object - self.position
            vector_to_object /= np.linalg.norm(vector_to_object)
            self.ground_velocity_vector = vector_to_object*np.linalg.norm(wind) 
        if self.behavior == 'land':
            self.ground_velocity_vector = 0
            print 'LANDED'
            return self.time

        # save history
        self.position = self.position + self.ground_velocity_vector*dt
        self.history_position.append(self.position)
        self.history_velocity.append(self.ground_velocity_vector)
        self.history_heading.append(get_heading(self.ground_velocity_vector, self.wind))
        self.history_odor.append(odor)
        self.history_wind.append(self.wind)
        
class World(object):
    def __init__(self, wind, n_visual_features, save_data=False):
        # world parameters, including wind, odor, and visual aspects
        self.wind = wind
        self.windmag = np.linalg.norm(wind)
        self.dt = 5
        self.time = 0
        self.odor_packet_radius = 30 # 3 cm
        self.n_odor_packets = 50
        self.odor_packets = []
        self.probability_of_new_packet_per_second = 0.005 # note time scale is running in milliseconds
        self.probability_of_new_packet = self.probability_of_new_packet_per_second*self.dt
        self.max_wind_rotation_angle_per_second = 100/1000.*np.pi/180. # 100 deg per second
        self.max_wind_rotation_angle = self.max_wind_rotation_angle_per_second*self.dt
        self.initialize_visual_features(n_visual_features)
        self.odorous_visual_feature = self.visual_features[0]
        
        # start with no flies
        self.fly = None
        
        # save odor data for plotting        
        self.save_data = save_data
        if self.save_data:
            visual_features = []
            for visual_feature in self.visual_features:
                pos = visual_feature.position
                if type(pos) is not list:
                    pos = (pos.T)[0].tolist()
                pos.append(visual_feature.radius)
                visual_features.append(pos)
            self.data = {'visual_features': visual_features, 'odor_packets': [[]], 'fly': [[]]}
            
    def run(self, i, max_time=10000, fly_parameters=None):
        
        self.time += self.dt
        
        # randomly make new odor packet
        r = np.random.random()
        if r < self.probability_of_new_packet:
            self.generate_odor_packet(self.odorous_visual_feature)

        # update wind and propagate odor packets
        self.update_wind()
        self.propogate_odor_packets()

        # once an odor packet is 1 meter from the source, create a fly inside that odor packet
        value = None
        if self.fly is None:
            if len(self.odor_packets) > 0:
                if np.linalg.norm(self.odor_packets[0].position) > 1000:
                    self.fly = Fly(self.odor_packets[0].position, fly_parameters=fly_parameters)
        else:            
            fly_odor = self.calculate_odor_for_fly()
            value = self.fly.update(fly_odor, self.wind, self.dt, self.visual_features)

        # save data for plotting
        if save_data:
            odor_packets = []
            for odor_packet in self.odor_packets:
                pos = odor_packet.position
                if type(pos) is not list:
                    pos  = (pos.T)[0].tolist()
                pos.append(odor_packet.radius)
                odor_packets.append(pos)
            self.data['odor_packets'].append(odor_packets)
            if self.fly is not None:
                self.data['fly'].append(self.fly.position)
            else:
                self.data['fly'].append([])
        if value is not None:
            if save_data:
                f = open('sim_data.pickle', 'w')
                pickle.dump(self.data, f)
                f.close()  
            return value

        # check end condition
        if self.fly is not None:
            if self.fly.time > max_time:
                return max_time
            else:
                return None
            
            
    ### Helper methods ###
    
    def calculate_odor_for_fly(self):
        if len(self.fly.history_position)<2:
            return False
        
        distances = []
        for odor_packet in self.odor_packets:
            # find closest point to odor packet center in range between flies position now, and last time (ie. integrate)
            d = dist_point_to_line(odor_packet.position, self.fly.history_position[-2], self.fly.history_position[-1], resolution=2)
            distances.append(d)
        index = np.argmin(distances)
        # best candidate
        d = dist_point_to_line(self.odor_packets[index].position, self.fly.history_position[-2], self.fly.history_position[-1], resolution=50)
        if d < odor_packet.radius:
            return True
        return False
            
    def initialize_visual_features(self, n_visual_features):
        self.visual_features = []
        for n in range(n_visual_features):
            if n==0:
                position = np.array([[0],[0],[0]])
                radius = 1
            else:
                position = (np.random.rand(3,1)-0.5)*2*self.scale*0.2
                radius = 1
            vf = VisualFeature(position, radius)
            self.visual_features.append(vf)
            
    def generate_odor_packet(self, visual_feature):
        radius = self.odor_packet_radius
        new_op = OdorPacket(visual_feature.position, radius)
        self.odor_packets.append(new_op)
        while len(self.odor_packets) > self.n_odor_packets:
            p = self.odor_packets.pop(0)
        
    def propogate_odor_packets(self):
        if len(self.odor_packets) is not None:
            for odor_packet in self.odor_packets:
                odor_packet.update(self.wind, self.dt)
        
    def update_wind(self):
        axis = np.random.rand(1,3)-np.array([.5,.5,.5])
        theta = 2*(np.random.rand()-0.5)*self.max_wind_rotation_angle
        self.wind = np.dot(rotation_matrix(axis,theta),self.wind)

################################################################################################   

        
if __name__ == '__main__':
    
    save_data = False

    # save data to make animations
    if save_data:
        world = World(np.array([[-.4], [0], [0]]), 1, save_data=True)
        time = None
        while time is None:
            fly_parameters = {'surge_delay': 270,
                              'cast_delay': 640,
                              'cast_time_amplitude': 500,
                              'altitude_time_amplitude': 309.0169,
                              'visual_attraction_probability': 1,
                              }
            time = world.run(0, max_time=20000, fly_parameters=fly_parameters)
        
        
    # run simulation for three different algorithms
    if save_data is False:
        
        niterations = 1000
        results = []
        # baseline
        baseline = []
        for iteration in range(niterations):
            world = World(np.array([[-.4], [0], [0]]), 1)
            time = None
            while time is None:
            
                fly_parameters = {'surge_delay': 270,
                                  'cast_delay': 640,
                                  'cast_time_amplitude': 500,
                                  'altitude_time_amplitude': 309.0169,
                                  'visual_attraction_probability': 1,
                                  }
                time = world.run(0, max_time=20000, fly_parameters=fly_parameters)
            print time
            baseline.append(time)
        results.append(baseline)
        
        # no vision
        baseline = []
        for iteration in range(niterations):
            world = World(np.array([[-.4], [0], [0]]), 1)
            time = None
            while time is None:
            
                fly_parameters = {'surge_delay': 270,
                                  'cast_delay': 640,
                                  'cast_time_amplitude': 500,
                                  'altitude_time_amplitude': 309.0169,
                                  'visual_attraction_probability': 0,
                                  }
                time = world.run(0, max_time=20000, fly_parameters=fly_parameters)
            print time
            baseline.append(time)
        results.append(baseline)
        
        # equal delay
        baseline = []
        for iteration in range(niterations):
            world = World(np.array([[-.4], [0], [0]]), 1)
            time = None
            while time is None:
            
                fly_parameters = {'surge_delay': 270,
                                  'cast_delay': 270,
                                  'cast_time_amplitude': 500,
                                  'altitude_time_amplitude': 309.0169,
                                  'visual_attraction_probability': 1,
                                  }
                time = world.run(0, max_time=20000, fly_parameters=fly_parameters)
            print time
            baseline.append(time)
        results.append(baseline)
        
        f = open('sim_results.pickle', 'w')
        pickle.dump(results, f)
        f.close()  

