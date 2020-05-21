#!/usr/bin/python3

# code for mean payoff games, deterministic mdps

from random import seed, randint, sample
import numpy as np
from fractions import Fraction # for exact calculation
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Circle
import matplotlib.patheffects as path_effects
import matplotlib.cm as cm
#import colorsys
import os
import sys


# useful classes


class state:

    def __init__(self, owner, next_states, cost):
        self.owner = owner
        self.next_states = next_states
        self.cost = cost

    def print(self):
        print("owner:",self.owner,"->",self.next_states,"c:",self.cost)



class mp_game:   

    # basic
    
    def __init__(self, nb_states, param, intmax=100):

        self.nb_states = nb_states
        if isinstance(param,list):    # param provides the model...
            self.states = param
        else:                         # ... or a number of actions per state
            nb_actions = param
            self.states = [ state( randint(0,1),   # owner
                                 [randint(0,nb_states-1) for a in range(nb_actions)],   # next states
                                 randint(0,intmax) )   # costs
                            for i in range(nb_states) ]

    def print(self):

        print(self.nb_states,"states")
        for i in range(self.nb_states):
            print(i,' ',end='')
            self.states[i].print()


    # policy analysis, policy iteration (1 and 2 players)
            
    def analyze_policy(self, pol, verbose=False): # returns paths and cycles for each starting state

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

                    if traj[j]==min(traj[j:]): # filter by cycles by begin with the minimal state (convention to avoid cycles)
                    
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


    def value_potential(self, pol): # compute the value and potential of a policy (useful for PI)

        v = []
        cycles, c_v, path, p_v, cycle = self.analyze_policy(pol)
        for i in range(self.nb_states):
            av = c_v[ cycle[i] ]
            v.append( (av, p_v[i] - av*len(path[i])) ) 
            #print('v(',i,')=',v[i])
            
        return(v)

    
    
    def pi_greedy_policy(self, v, pol, player):
        
        pol2 = pol[:] # copy

        for i in range(self.nb_states):
            state = self.states[i]
            if state.owner==player: 
                aopt = pol[i]   # initialise with current policy
                qopt = v[ state.next_states[aopt] ]  # and value
                for a in range(len(state.next_states)):
                    q = v[ state.next_states[a] ]
                    if (player==0 and (q[0]<qopt[0] or (q[0]==qopt[0] and q[1]<qopt[1]))) or (player==1 and (q[0]>qopt[0] or (q[0]==qopt[0] and q[1]>qopt[1]))):
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
            pol = self.pi_greedy_policy(v, pol, player)
            
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
            pol = self.pi_greedy_policy(v, pol, player)
            
            v2 = v

        return v, pol

    
    # value iteration
    
    def value_iteration(self, T, v=[]):
        
        if v==[]:
            v = [0]*self.nb_states

        pol_seq = []
        v_seq = []
        
        for t in range(T-1,-1,-1):

            pol = []
            v2 = []
            
            for i in range(self.nb_states):

                state = self.states[i]
                owner = state.owner

                lv =  [ v[ state.next_states[a] ]  for a in range(len(state.next_states)) ]
                if state.owner == 0:
                    qopt = min( lv )
                elif state.owner == 1:
                    qopt = max( lv )
                    
                pol.append( lv.index(qopt) )
                v2.append( state.cost + qopt )
                
            v_seq.append(v2)
            pol_seq.append(pol)

            v = v2

        pol_seq.reverse() # put it in the right order
        
        return v_seq, pol_seq  


    def trajectory(self, i, pol_seq): # generate a trajectory from finite horizon policy pol_seq

        traj = [i]
        for t in range(len(pol_seq)): # backward in time
            i = self.states[i].next_states[ pol_seq[t][i] ]
            traj.append(i)
        return traj
        

    def get_ax(self):

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.axis('off')
        return(ax)

    def savefig_and_close(self,f):
        
        plt.savefig(f)
        plt.close()

        
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
            
    def plot_graph(self, ax, pol): # plot a stationary policy

        lx,ly=self.range

        RAYON=0.25
        plt.xlim(-1,lx)
        plt.ylim(-1,ly)

        for i in range(self.nb_states):
            x,y = self.pos[i]
            state = self.states[i]
            if state.owner==0:
                ax.add_patch(Circle( (x,y), RAYON, fill=0, edgecolor='black') )
            else:
                ax.add_patch(Rectangle( (x-RAYON,y-RAYON), 2*RAYON,2*RAYON, fill=0, edgecolor='black') )
            ax.text(x,y,str(state.cost),va='center',ha='center')
            for j in state.next_states:
                state2 = self.states[j]
                x2,y2 = self.pos[j]
                dx,dy=x2-x,y2-y
                if pol!=None and state.next_states[ pol[i] ]==j:
                    plt.arrow( x+RAYON*dx, y+RAYON*dy, dx/2.5, dy/2.5, color='black',lw=1, head_width=0.1,zorder=1)
                else:
                    plt.arrow( x+RAYON*dx, y+RAYON*dy, dx/2.5, dy/2.5, color='black', alpha=0.1, lw=1, head_width=0.1,zorder=0)

        plt.tight_layout()
                    
    def plot_trajectory(self, ax, traj, T=None, color='black'):

        l = len(traj)
        if T==None:
            T=l
        
        lx,ly = [],[]
        dec = 2*np.pi/T
        for t in range(l):
            x,y = self.pos[ traj[t] ]
            angle = (T-t)*dec
            lx.append(x+0.25*np.cos(angle))
            ly.append(y+0.25*np.sin(angle))

        plt.plot(lx,ly, color=color, lw=2, path_effects=[path_effects.Stroke(linewidth=3, foreground='black'), path_effects.Normal()])

        plt.tight_layout()
                
def planar_mp_game(lx, ly, nb_actions, intmax=100):

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
            param.append( state( randint(0,1),  sample(neighbors, min(nb_actions,len(neighbors))),  pow(2,randint(0,intmax))  ) )

    g =  mp_game(lx*ly, param)
    g.pos=dict()
    for x in range(lx):
        for y in range(ly):
            g.pos[x*ly+y]=(x,y)
    g.range=(lx,ly)
            
    return ( g )
        


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



def test_1():

    for i in range(0,10000):
        seed(i)
        g = mp_game(20,4,100)
        #g.print()
        v,pol = g.policy_iteration(player=1)
        cycles, c_v, path, p_v, cycle = g.analyze_policy( pol )
        if len(cycles)>5:
            print(i)
            g.print()
            print('policy:',pol)
            cycles, c_v, path, p_v, cycle = g.analyze_policy( pol, verbose=True )


def test_2():

    x,y = 4,4   
    max_len=0

    os.system('rm pi/*')
    
    for i in range(10000):
        print(i,end='\r')
        seed(i)
        g=planar_mp_game(x,y,3,20)
        policy_list = []
        v,pol = g.policy_iteration(player=1,  policy_list=policy_list)

        if len(policy_list) >= max_len:
            max_len = len(policy_list)
            cycles, c_v, path, p_v, cycle = g.analyze_policy( pol )
            print('PI visited',len(policy_list),'policies (',len(cycles),'cycles (seed=',i,')')
            print('Saves:',end='')
            j=0
            for pol in policy_list:
                print(j,end=',')
                sys.stdout.flush()
                cycles, c_v, path, p_v, cycle = g.analyze_policy( pol )
                ax = g.get_ax()
                g.plot_cycle_regions(ax, cycles,cycle,c_v)
                g.plot_graph(ax, pol)
                g.savefig_and_close("pi/%d_%03d.png"%(i,j))
                j+=1
            print()


def test_3():
    
    X = True

    for sd in [2]:#range(100):

        x,y = 5,5
        seed(sd)
        print('seed:',sd)

        g=planar_mp_game(x,y,3,20)

        # résolution exacte par PI et détermination des cycles optimaux
        v,pol = g.policy_iteration(player=1)
        cycles, c_v, path, p_v, cycle = g.analyze_policy( pol )


        if X:
            os.system('rm vi/*')
            ax = g.get_ax()
            g.plot_cycle_regions(ax, cycles, cycle, c_v)
            g.plot_graph(ax, pol)
            g.savefig_and_close("vi/opt.png")

        # résolution par VI
        T = 1000
        v_seq, pol_seq = g.value_iteration(T)

        if X:
            ord = np.argsort(c_v)
            cmap = cm.get_cmap('autumn')
            if len(cycles)==1:
                colors = [ cmap(0.5) ]
            else:
                colors = [ cmap( float(ord[i])/(len(cycles)-1) )  for i in range(len(cycles)) ]


        for c in cycles:
            i=c[0]
            traj = g.trajectory(i, pol_seq )
            if traj[:len(c)]!=c:
                print(traj,c)    ############################## faire de l'analyse de chemin...

        if X:

            for t in range(1,T):

                print(t,',',end='')
                sys.stdout.flush()
                
                ax = g.get_ax()
                g.plot_cycle_regions(ax, cycles, cycle, c_v)

                for j in range(len(cycles)):
                    i=cycles[j][0]
                    traj = g.trajectory(i, pol_seq[T-t:])
                    g.plot_trajectory(ax, traj, T=T, color=colors[j])
                    g.plot_graph(ax, pol_seq[T-t])

                plt.savefig("vi/%03d.png"%t)
                plt.close()




test_3()
