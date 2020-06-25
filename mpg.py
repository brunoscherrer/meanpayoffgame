#!/usr/bin/python3

# code for undiscounted mean payoff games

from copy import copy,deepcopy
import os
import sys

from random import seed, randint, sample
import numpy as np
from fractions import Fraction # for exact calculation
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
import matplotlib.patheffects as path_effects
import matplotlib.cm as cm

import networkx as nx # graph display (to illustrate & debug)



# undiscounted mean-payoff game model

class state:

    def __init__(self, player, next_states, cost, id):
        self.player = player
        self.next_states = next_states
        self.cost = cost
        self.id = id

    def print(self, end=''):
        print("id:", self.id, "player:",self.player,['(MIN)','(MAX)'][self.player],"next_states:",self.next_states,"c:",self.cost, end=end)


class mp_game:   

    # basic
    
    def __init__(self, nb_states, param, intmax=100):

        self.nb_states = nb_states
        if isinstance(param,list):    # param provides the model (as a list)
            self.states = param
        else:                         # ... or a number of actions per state
            nb_actions = param
            self.states = [ state( randint(0,1),   # player
                                 [randint(0,nb_states-1) for a in range(nb_actions)],   # next states
                                   randint(0,intmax),    # costs
                                   str(i) ) # id
                            for i in range(nb_states) ]

    def print(self):

        print(self.nb_states,"states")
        for i in range(self.nb_states):
            print("State",i,end=": ")
            self.states[i].print(end="/ ")
            print( [ self.states[k].id for k in self.states[i].next_states ] )



    #####
    # Comparison of value potential

    @staticmethod
    def better( v1, v2, player ):   # tells if the value-potential v1 is *strictly* better than v2 for Player player
        if player==0: # min
            return (v1[0]!=None and v2[0]==None) or (v1[0]!=None and v2[0]!=None and v1[0]<v2[0]) or (v1[0]==v2[0] and v1[1]<v2[1]) 
        else: # max
            return (v1[0]!=None and v2[0]==None) or (v1[0]!=None and v2[0]!=None and v1[0]>v2[0]) or (v1[0]==v2[0] and v1[1]>v2[1])


    #####
    # (standard) Value Iteration (on value potential)
    
    def value_iteration(self, T, v=None, verbose=False):

        if v==None:
            v = [(None,0)]*self.nb_states  # initialize to value-potential
        
        pol_seq = []
        v_seq = []
        
        for t in range(T-1,-1,-1):

            pol = []
            v2 = []
            
            for i in range(self.nb_states):

                s = self.states[i]
                player = s.player

                qopt = [(None, np.infty),(None, -np.infty)][player]
                aopt=-1
                for a in range(len(s.next_states)):
                    q =  v[ s.next_states[a] ] 
                    if self.better(q,qopt,player):
                        qopt,aopt = q,a
                        
                pol.append( aopt )
                if qopt[0]!=None:
                    v2.append( ( qopt[0], s.cost-qopt[0] + qopt[1]) )
                else:
                    v2.append( ( None, s.cost + qopt[1] ) )

            if verbose:
                print("pol=",pol)
                print("v=",pvp(v))
                    
            v_seq.insert(0,v2)
            pol_seq.insert(0,pol)

            v = v2
        
        return v_seq, pol_seq  


    ######
    # (standard) Policy Iteration

    def analyze_policy(self, pol, verbose=False): # Find paths and cycles of a joint policy pol (used by policy iteration to compute the value)

        cycles = [] # list of cycles

        c_v = []
        nb_cycles=0
        
        path = [0]*self.nb_states    # paths for all starting states
        p_v = [0]*self.nb_states     # value on the path
        cycle = [0]*self.nb_states   # cycle number reached

        while 0 in path: # while starting state does not know path/cycle/etc...
            
            i = path.index(0)
            traj = [i]
            costs = [self.states[i].cost]

            while True:
                
                # make one step
                x = self.states[i]
                a = pol[i]
                i = x.next_states[a]
                
                if type(path[i]) is list:   # we connect to an already existing path-cycle
                    
                    for n in range(len(traj)):
                        k = traj[n]
                        path[ k ] =  traj[ n: ] + path[ i ]
                        p_v[ k ] =  sum(costs[ n:] ) + p_v[ i ]
                        cycle[ k ] = cycle[ i ]
                    break
 
                elif i in traj: # we discover a new cycle

                    # extract 
                    j = traj.index(i) # beginning index of the cycle
                    c = traj[j:]

                    if traj[j]==min(traj[j:]): # filter by cycles that begin with the minimal state (convention)
                    
                        v = Fraction( sum(costs[j:]) , (len(traj)-j ) ) 
                        cycles.append(c)
                        c_v.append(v)
                        for n in range(len(traj)):
                            k = traj[n]
                            if n<=j:
                                path[ k ] =  traj[ n:j ]
                                p_v[ k ] =  sum(costs[ n:j ])
                                cycle[ k ] = nb_cycles
                            else:
                                path[ k ] =  traj[ n: ]
                                p_v[ k ] =  sum(costs[ n: ])
                                cycle[ k ] = nb_cycles        
                        nb_cycles+=1
                        break

                costs.append( self.states[i].cost )  
                traj.append(i)
        
        if verbose:
            for i in range(len(cycles)):
                print("cycle",i,":", cycles[i], c_v[i])
            for i in range(self.nb_states):
                print('state',i,'(cost,',self.states[i].cost,')',path[i],p_v[i],'-> (',cycles[cycle[i]],c_v[cycle[i]],')')
                
        return cycles, c_v, path, p_v, cycle

    
    def value_potential(self, pol): # compute the value and potential of a joint policy pol

        v = []
        cycles, c_v, path, p_v, cycle = self.analyze_policy(pol)
        for i in range(self.nb_states):
            av = c_v[ cycle[i] ]
            v.append( (av, p_v[i] - av*len(path[i])) ) 
            
        return v
    
    
    def greedy_policy(self, v, pol, players):
        
        pol2 = pol[:] # copy

        for i in range(self.nb_states):
            s = self.states[i]
            player = s.player
            if player in players:
                aopt = pol[i]   # initialise with current policy
                qopt = v[ s.next_states[aopt] ]  # and value
                for a in range(len(s.next_states)):
                    q = v[ s.next_states[a] ]
                    if self.better(q, qopt, player):
                        qopt, aopt = q, a
                pol2[i] = aopt
                    
        return pol2
    
    
    def one_player_policy_iteration(self, player=1, pol=[], policy_list=None):
        
        if pol==[]:
            pol = [0]*self.nb_states

        v2 = None
        
        while True:

            # value estimation
            v = self.value_potential(pol)
                
            # test for stopping
            if v==v2:
                break

            if policy_list!=None:
                policy_list.append(pol)
            
            # greedy step (for the states of the player who optimizes)
            pol = self.greedy_policy(v, pol, [player])
            
            v2 = v

        return v, pol
    

    def policy_iteration(self, player=1, pol=[], policy_list=None):

        if pol==[]:
            pol = [0]*self.nb_states

        v2 = None
        
        while True:
            
            # value estimation (optimize for the other player)
            v, pol = self.one_player_policy_iteration(1-player, pol, policy_list=policy_list)
            
            # test for stopping
            if v==v2:
                break
            
            # greedy step (for the states of the player who optimizes)
            pol = self.greedy_policy(v, pol, [player])
            
            v2 = v

        return v, pol

    
    #####################
    ### The algorithm

    def trajectory(self, i, pol_seq): # generate a trajectory from finite horizon policy pol_seq 

        traj = [i]
        for t in range(len(pol_seq)): # backward in time
            i = self.states[i].next_states[ pol_seq[t][i] ]
            traj.append(i)
        return traj


    def cycle_id(self, cycle):

        s=""
        for i in cycle[:-1]:
            id = self.states[i].id
            s += id+"-"
        s += self.states[cycle[-1]].id
        return s
            
    
    def algo(self, verbose=False):

        # initialize
        nb_states = self.nb_states
        states_list = [ [i] for i in range(nb_states) ] 
        real_states = list(range(nb_states))
        
        # Initialization with the cycles of any policy
        pol = [0]*self.nb_states
        cycles, av_costs, _,_,_ = self.analyze_policy(pol,verbose=verbose)
        for i in range(len(cycles)):
            cycles[i].append(cycles[i][0]) # close the loop
        if verbose: print("Initialization with policy:",pol,"\n")

        cycle_list, av_costs_list = [],[]
        
        while(True):

            # Part 1: make the game from the list of states (determining the proper transition regarding memory)
            
            cycle_list += cycles
            av_costs_list += av_costs
            
            for i in range(len(cycles)):

                c = cycles[i]
                
                # add cycle to model
                if verbose: print("=> Found new cycle",c,"with average cost", av_costs[i])
                        
                nb_states += 1
                states_list.append(c)
                real_states.append(None) # cycle does not correspond to any ground state 

                # add paths-to-cycle to model (if necessary)
                for j in range(len(c)-1,0,-1):

                    c2 = c[:j]
                    if c2 not in states_list:
                        nb_states += 1
                        states_list.append(c2)
                        real_states.append(c2[-1])
            
            if verbose:
                print("List of cycles found so far:",cycle_list)
                print("Average costs:",av_costs_list)
            
            # build the model
            param = []
            for i in range(nb_states):
                
                if real_states[i]==None:      # if terminal state-cycle
                    j = cycle_list.index( states_list[i] )
                    param.append( state( 0, [i], av_costs_list[j], self.cycle_id( states_list[i] ) )  )

                else: # usual states
                    s = self.states[ real_states[i] ]
                    ns = []
                    for a in range(len(s.next_states)):       # find 
                        s2 = states_list[i]+[ s.next_states[a] ]
                        while True:
                            if s2 in states_list:
                                ns.append( states_list.index(s2) )
                                break
                            del s2[0]
                    param.append( state( s.player, ns, s.cost, self.cycle_id( states_list[i] ) ) )

            m = mp_game( nb_states, param ) 

            if verbose:
                m.print()
                print("List of states:",states_list)
                m.plot_graph(get_ax())
                plt.show()


            # Part 2: find new cycles and add them
            
            cycles, av_costs, pol_seq, v = self.find_new_cycles(m, self.nb_states, states_list, real_states, cycle_list, av_costs_list, verbose=verbose )

            if cycles==[]:
                break

        
        print("List of cycles used") # sort list by starting state
        for i in range(self.nb_states):
            j=0
            for c in cycle_list:
                if c[0]==i:
                    print(c, '(', av_costs_list[j], ')', end=' ')
                j+=1
            print('')


        #exit(1)
            
        return(v[0:self.nb_states],pol_seq[0][0:self.nb_states])
    

    # procedure to find cycles of length c involving x

    def find_new_cycles(self, m, N, states_list, real_states, cycle_list, av_costs_list, verbose=False): # N is the number of real states

        # initialize to value-potential
        v = []
        for i in range(m.nb_states): #####
            if real_states[i] == None:  # if state is a cycle
                sl = states_list[i]
                j = cycle_list.index( sl ) # cycle_list[j] = sl
                k = sl.index(min(sl))      # position of the state with minimal index (convention)
                print('*** sl=',sl,'j=',j, 'k=',k,  av_costs_list[j], [ self.states[sl[l]].cost - av_costs_list[j] for l in range(k) ])
                v.append(   ( av_costs_list[j], sum([ self.states[sl[l]].cost - av_costs_list[j] for l in range(k) ]) )  )  # value, potential
            else:
                v.append( (None,0) )
                
        cycles = []
        av_costs = []
        
        pol_seq = []

        if verbose:
            print("* Find new cycles from terminal value\nv=", pvp(v) )
        
        for t in range(N+1): # N+1 to be able to cycle before entering a cycle 

            pol = []
            v2 = []
            
            for i in range(m.nb_states):  # VI-step

                s = m.states[ i ]
                player = s.player

                qopt = [(None, np.infty),(None, -np.infty)][player]
                aopt=-1
                for a in range(len(s.next_states)):
                    q =  v[ s.next_states[a] ] 
                    if self.better(q,qopt,player):
                        qopt,aopt = q,a
                        
                pol.append( aopt )
                if qopt[0]!=None:
                    v2.append( ( qopt[0], s.cost-qopt[0] + qopt[1]) )
                else:
                    v2.append( ( None, s.cost + qopt[1] ) )

            if verbose:
                #print("pol=",pol)
                print("v=",pvp(v))

            pol_seq.insert(0,pol)
                
            v = v2
            
            for i in range(m.nb_states):   # m, not self!

                if real_states[i] != None: # we do not start in a cyle
                
                    traj = m.trajectory(i, pol_seq)

                    if verbose: print(traj,end=" / ")
                    
                    traj = [ real_states[x] for x in traj ] # get trajectory on ground states
                
                    if verbose: print(traj)
                
                    if traj[0] in traj[1:]:                     # detect cyle
                        s = traj[1:].index(traj[0]) + 1         
                        c = traj[:s+1]                    # corresponding cycle
                        av_cost = Fraction( sum( [self.states[traj[j]].cost for j in range(s)] ), s )
                        if c not in cycle_list and c not in cycles:  # new cycle ?
                            cycles.append(traj[:s+1])
                            av_costs.append( av_cost )

            if cycles!=[]: # leaves as soon as at least one new cycle is found
                break

        return cycles, av_costs, pol_seq, v


    
    def plot_graph(self, ax, pol=None): # plot a stationary policy
            
        NS=1000
        
        g = nx.DiGraph()
        labels={}
        
        for i in range(self.nb_states):
            s = self.states[i]
            id = s.id
            labels[id]=id
            for j in s.next_states:
                s2 = self.states[j]
                g.add_edge(id, s2.id)
                

        # pos = nx.spring_layout(g, pos=pos, fixed=fixed, iterations=50)
        pos = nx.kamada_kawai_layout(g)

        nodes = nx.draw_networkx_nodes(g, pos,  ax=ax, nodelist=[ x.id for x in self.states if x.player==0 ], node_size=NS, node_shape='o', alpha=1, node_color='w')
        nodes.set_edgecolor('k')
        nodes = nx.draw_networkx_nodes(g, pos,  ax=ax, nodelist=[ x.id for x in self.states if x.player==1 ], node_size=NS, node_shape='s', alpha=1, node_color='w')
        nodes.set_edgecolor('k')
        nx.draw_networkx_edges(g, pos,  ax=ax, node_size=NS, edge_color='lightgrey')
        if pol!=None:
            nx.draw_networkx_edges(g, pos,  ax=ax, edgelist=[ (self.states[i].id, self.states[ self.states[i].next_states[pol[i]] ].id) for i in range(self.nb_states) ], node_size=NS, edge_color='k')
        nx.draw_networkx_labels(g, pos, labels,  ax = ax, font_size=NS/150)

        plt.tight_layout()
    
        
### Mean payoff game on a grid structure


class planar_mp_game(mp_game):

    def __init__(self, lx, ly, nb_actions, intmax=100):

        param = []
        for x in range(lx):
            for y in range(ly):
                neighbors=[]
                for dx in [-1,0,1]:
                    for dy in [-1,0,1]:
                        x2,y2 = x+dx,y+dy
                        if (dx,dy)!=(0,0) and 0<=x2<lx and 0<=y2<ly:
                            neighbors.append(x2*ly+y2)
                i=x*ly+y
                param.append( state( randint(0,1),  sample(neighbors, min(nb_actions,len(neighbors))),  pow(2,randint(0,intmax)), str(i)  ) )

        super().__init__(lx*ly, param)

        # for graphical display
        
        self.pos=dict()
        for x in range(lx):
            for y in range(ly):
                self.pos[x*ly+y]=(x,y)    
        self.range=(lx,ly)

        
    def plot_cycle_regions(self, ax, cycles,cycle,c_v):

        lx,ly=self.range        

        ord = np.argsort(c_v)
        cmap = cm.get_cmap('autumn')
        if len(cycles)==1:
            colors = [cmap(0.5)]
        else:
            colors = [ cmap( float(ord[i])/(len(cycles)-1) )  for i in range(len(cycles)) ] 
        
        for i in range(self.nb_states):
            x,y = self.pos[i]
            if i in cycles[cycle[i]]:
                a=0.4
            else:
                a=0.2
            ax.add_patch(Rectangle( (x-.5,y-.5),1,1, color=colors[cycle[i]], alpha=a, linewidth=0) )

        plt.tight_layout()
            
    def plot_graph(self, ax, pol=None): # plot a stationary policy

        lx,ly=self.range

        RAYON=0.25
        plt.xlim(-1,lx)
        plt.ylim(-1,ly)

        for i in range(self.nb_states):
            x,y = self.pos[i]
            s = self.states[i]
            if s.player==0:
                ax.add_patch(Circle( (x,y), RAYON, fill=0, edgecolor='black') )
            else:
                ax.add_patch(Rectangle( (x-RAYON,y-RAYON), 2*RAYON,2*RAYON, fill=0, edgecolor='black') )
            ax.text(x,y,str(s.cost),va='center',ha='center')
            for j in s.next_states:
                x2,y2 = self.pos[j]
                dx,dy = x2-x,y2-y
                col = ['blue','red'][s.player]
                r=RAYON/math.sqrt(dx*dx+dy*dy)
                if pol!=None and s.next_states[ pol[i] ]==j:
                    plt.arrow( x+r*dx, y+r*dy, dx*(1-2.7*r), dy*(1-2.7*r),lw=1, head_width=0.1,zorder=1, color=col)
                else:
                    plt.arrow( x+r*dx, y+r*dy, dx/2.5, dy/2.5, alpha=0.1, lw=1, head_width=0.1,zorder=0, color=col)

        plt.tight_layout()

        
    def plot_trajectory(self, ax, traj, T=None, color='black'):

        RAYON=0.1 #0.25
        
        l = len(traj)
        if T==None:
            T=l
        
        lx,ly = [],[]
        dec = 2*np.pi/T
        for t in range(l):
            x,y = self.pos[ traj[t] ]
            angle = (T-t)*dec
            lx.append(x+RAYON*np.cos(angle))
            ly.append(y+RAYON*np.sin(angle))
            if t>0:
                plt.arrow( lx[-2],ly[-2],(lx[-1]-lx[-2])*0.9,(ly[-1]-ly[-2])*0.9, head_width=0.1, color=color, lw=2 )
        
        #plt.plot(lx,ly, color=color, lw=2, path_effects=[path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

        plt.tight_layout()




def parity_game(nb_states, nb_actions, nb_priorities):

    g=mp_game(2,nb_states, nb_actions)
    for i in range(nb_states):
        for a in range(nb_actions):
            p=randint(0,nb_priorities-1)
            if p%2==0:
                g.states[i].costs[0, a] = pow(nb_states,p)
            else:
                g.states[i].costs[0, a] = -pow(nb_states,p)
            g.states[i].costs[1, a] = -g.states[i].costs[0, a]
            
    return g

##############################################################




def pvp(v): # printable value-potential
    return list(map(lambda x:(float(x[0]) if x[0]!=None else None,float(x[1])),v))        


def get_ax():

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    ax.axis('off')
    return(ax)

def savefig_and_close(f):
        
    plt.savefig(f)
    plt.close()
