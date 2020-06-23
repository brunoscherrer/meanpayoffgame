#!/usr/bin/python3

from mpg import *

x,y = 4,4

for i in range(10,1000): # (3,3,22, 2,2,0)

    print('Seed:',i)
    seed(i)
    g=planar_mp_game(x,y,3,20)
    g.print()

    # solve by PI
    v,pol = g.policy_iteration(player=1)
    
    # algorithm
    v2,pol2 = g.algo(verbose=True)

    if True:#pol!=pol2:

        print(pol)
        print(pol2)
        
        cycles, c_v, path, p_v, cycle = g.analyze_policy( pol )
        ax = get_ax()
        g.plot_cycle_regions(ax, cycles,cycle,c_v)
        g.plot_graph(ax, pol)
        
        cycles, c_v, path, p_v, cycle = g.analyze_policy( pol2 )
        ax = get_ax()
        g.plot_cycle_regions(ax, cycles,cycle,c_v)
        g.plot_graph(ax, pol2)

        plt.show()
    

