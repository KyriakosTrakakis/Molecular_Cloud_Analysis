
# coding: utf-8

#MST algorithm

#usage

# alter value of l1,l2 to set Galactic longitude range
# alter value of my_Dl, my_DV to set critical separation of two connected vertex
# type filename containing Galactic longitude, latitude and velocity

l1,l2 = 250, 280

my_dL = .1
my_dV = 1
filename = 'mopra_higal_lbv_results.csv'

import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from matplotlib.cbook import flatten
import pickle
from matplotlib.ticker import MultipleLocator, FuncFormatter
import os

def bgps(l1=250, l2=280):
    import numpy as np
    from astropy import units as u

    fp = open(filename) 
    l=[]
    b=[]
    v=[]
    for linea in fp.readlines():
        linea = linea.strip().split(",")[:]
        try:
            v.append(float(linea[1]))
        except ValueError:
            continue
        l.append(float(linea[0][7:14]))
        if l[len(l)-1]>180:
            l[len(l)-1] -= 360
        b.append(float(linea[0][15:22])) 
    l = np.array(l)
    b = np.array(b)
    v = np.array(v)     
     
    # mask arrays
    # mask on vlsr
    mv = np.ma.masked_where(v<999, v).mask
    # mask on l
    ml = np.ma.masked_where( (l>l1)&(l<l2), l).mask
    mask = ml*mv
    l = l[mask]
    b = b[mask]
    v = v[mask]
     
    # form 3d and 2d arrays, to be used to generate MST
    lbv = np.zeros( (len(l), 3) )
    lbv[:,0] = l
    lbv[:,1] = b
    lbv[:,2] = v
    lb = lbv[:,0:2]
 
    return lbv
  
def minimum_spanning_tree(X, v, copy_X=True, my_dL=my_dL):
    #X are edge weights of fully connected graph
    if copy_X:
        X = X.copy()
  
    if X.shape[0] != X.shape[1]:
        raise ValueError("X needs to be square matrix of edge weights")
    n_vertices = X.shape[0]
    spanning_edges = []
    break_point = []
      
    # initialize with node 0:                                                                                         
    visited_vertices = [0]                                                                                            
    num_visited = 0
    # exclude self connections:
    diag_indices = np.arange(n_vertices)
    X[diag_indices, diag_indices] = np.inf
      
    while num_visited != n_vertices:
        new_edge = np.argmin(X[visited_vertices], axis=None) 
        # 2d encoding of new_edge from flat, get correct indices
        new_edge = divmod(new_edge, n_vertices) 
        new_edge = [visited_vertices[new_edge[0]], new_edge[1]]
         
        # add edge to tree
        edge_length = np.min(X[visited_vertices])
        dv = abs(v[new_edge[1]] - v[new_edge[0]])
         
        # define conditions to accept edge
        separation = (edge_length < my_dL)
 
        if True:
            if separation:
                spanning_edges.append(new_edge)
            else:
                break_point.append(len(spanning_edges))
        visited_vertices.append(new_edge[1])
        # remove all edges inside current tree
        X[visited_vertices, new_edge[1]] = np.inf
        X[new_edge[1], visited_vertices] = np.inf                                                                     
        num_visited += 1
 
    # now we have a list of spanning_edges, break the list into individual branches
    # build up branch list
    b = break_point
    b = [0]+b # index of first edge element
    b = b+[-1] # index of last edge element
    branch_list = []
    for i in range(0,len(b)-1):
        branch = spanning_edges[ b[i]:b[i+1] ]
        if len(branch) > 4:
            branch_list.append(branch)
    return branch_list
  
 
def test_mst(l1,l2, my_dV=my_dV):
    lbv = bgps(l1=l1, l2=l2)
    l,b,v = lbv[:,0], lbv[:,1], lbv[:,2]
    P = lbv[:,0:2] # 2d position l,b
    plt.close('all') # close all figures
    plt.figure(num=1, dpi=100, figsize=(16, 6)) #when you need to show a slice
    plt.scatter(P[:, 0], P[:, 1], s=100, c=v, cmap='cubehelix', alpha=.8)
    cb = plt.colorbar()
    cb.set_label(label=r'$V_\mathrm{LSR} \,\, (\mathrm{km \, s^{-1}})$', fontsize='x-large')
    cb.ax.tick_params(labelsize='x-large')
    plt.xlim(l2, l1)
    plt.xlabel('Galactic Longitude (deg)',fontsize='x-large')
    plt.ylabel('Galactic Latitude (deg)',fontsize='x-large')
    plt.tick_params(labelsize='x-large')
 
 
  
    X = squareform(pdist(P))
  #  max_length = np.percentile(pdist(P), 25) 
 
    # weight distance with velocity difference
    for i in range(0,X.shape[0]):
        for j in range(0,X.shape[0]):
            dv = abs(v[i] - v[j])
            if (dv > my_dV) & (X[i,j] < 1*my_dL):
                X[i,j] = X[i,j]*dv                 

    branch_list = minimum_spanning_tree(X,v)
    with open("branch_list.dat", 'wb') as f:
        pickle.dump(branch_list, f)
    with open("branch_list.dat", 'rb') as f:
        my_list = pickle.load(f)
 

    # per filament
    for branch in branch_list:
        if len(branch)>1: 
            branch = np.vstack(branch)
        print ('plotting branch has # of clumps:', len(branch)+1)
        print (branch)
 
        # all clumps in this filament
        clump_index = np.unique(np.array(branch))
        ci = clump_index
        l_wt = np.sum( l[ci] * v[ci] ) / np.sum(v[ci])
        b_wt = np.sum( b[ci] * v[ci] ) / np.sum(v[ci])
 
        for edge in branch:
            i, j = edge
            plt.plot([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]], c='k', lw=1, alpha=.6)
 
    print ('len(branch_list)', len(branch_list))
 
    plt.tight_layout()
  
if __name__ == "__main__":

    
    test_mst(l1, l2)

    ax = plt.gca()
    ax.xaxis.set_major_locator( MultipleLocator(0.5) )
    ax.yaxis.set_major_locator( MultipleLocator(0.2) )
    plt.tick_params(labelsize=10)
    plt.title(r'MST',fontsize='xx-large')
    plt.tight_layout()
    if not os.path.isdir('fig'):
        os.makedirs('fig')
    plt.savefig('fig/demo.eps')
    plt.savefig('fig/demo.pdf')
    plt.savefig('fig/demo.png')


filename = 'mopra_higal_lbv_results.csv'
fp = open(filename) 
for linea in fp.readlines():
    linea = linea.strip().split(",")[:]
    print(linea[0][4:11])