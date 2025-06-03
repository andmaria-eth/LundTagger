import numpy as np
import torch
import networkx as nx
from torch_geometric.utils import to_networkx
from torch_geometric.transforms import BaseTransform
from numpy import random
import matplotlib.pyplot as plt
import torch

class OnTheFlyNormalizer(BaseTransform):
    """
    Normalize the data on the fly by simple subtraction and division.

    Args:
        attrs: list of attributes to normalize
        means: means of the attributes
        stds: standard deviations of the attributes
    """
    def __init__(self, attrs: list, means: torch.tensor, stds: torch.tensor, device=None):
        self.attrs = attrs
        self.means = means.to(device) if device is not None else means
        self.stds = stds.to(device) if device is not None else stds
        
    def __call__(self, data):
        for store in data.stores:
            for key, value in store.items(*self.attrs):
                value -= self.means
                value /= self.stds
                store[key] = value

        return data

# Define binary tree structure for Lund decomposition
class Node:
    def __init__(self, value, index):
        self.value = value
        self.left = None
        self.right = None
        self.idx = index

def tensor_to_tree(nodes, edges):
    """
    Convert a binary tree from arrays to a tree structure for easier pruning.
    
    Args:
        nodes: array of node features
        edges: array of edges in the tree
        
    Returns:
        top_node: the top node of the tree
    """

    nodes = np.array(nodes)
    edges = np.array(edges)

    n_nodes = len(nodes)

    # Convert arrays into a binary tree representation for easier pruning
    # Find the top node of the tree
    top_node_index = np.max(edges[0])
    top_node = Node(nodes[top_node_index], top_node_index)

    # Start walking down the tree
    buffer = [top_node]
    visited_nodes = [top_node_index]
    while len(buffer) > 0:
        # Get the next node
        node = buffer.pop(0)
        visited_nodes.append(node.idx)

        # Find the children of the node
        node_children = [idx for idx in edges[1][np.where(edges[0] == node.idx)] if idx not in visited_nodes]

        # Check that the tree is binary as it should be
        if len(node_children) not in [0, 2]:
            raise ValueError("Lund Tree is not binary!")
        # Add the children to the tree
        if len(node_children) == 2:
            node.left = Node(nodes[node_children[0]], node_children[0])
            node.right = Node(nodes[node_children[1]], node_children[1])

            # Add the children to the buffer
            buffer.append(node.left)
            buffer.append(node.right)

    return top_node

def tree_to_tensor(top_node, shape):
        # Walk along the tree and convert it back into arrays
        nodes = np.zeros(shape)
        edges = [[], []]
        buffer = [top_node]
        top_node.idx = 0
        counter = 0
        nodes[0] = top_node.value

        while len(buffer) > 0:
            # Get the next node
            node = buffer.pop()
    
            # Add the children to the buffer
            if node.left is not None and node.right is not None:
                # Keep track of the global node index
                counter += 2
                node.left.idx = counter - 1
                node.right.idx = counter
                
                # Add the childrens features to the nodes array
                nodes[node.left.idx] = node.left.value
                nodes[node.right.idx] = node.right.value

                # Add the children to the buffer
                buffer.append(node.left)
                buffer.append(node.right)

                # Add the edges to the edges array
                edges[0].append(node.idx)
                edges[1].append(node.left.idx)
                edges[0].append(node.idx)
                edges[1].append(node.right.idx)
                edges[1].append(node.idx)
                edges[0].append(node.left.idx)
                edges[1].append(node.idx)
                edges[0].append(node.right.idx)
    
        edges = np.array(edges)
        nodes = np.array(nodes)
    
        return torch.tensor(nodes, dtype=torch.float), torch.tensor(edges, dtype=torch.long)

def prune_tree(top_node, cut_variable_idx, cut_value):

    # Walk along the tree and prune nodes that do not pass the cut
    buffer = [top_node]
    n_nodes = 0
    while len(buffer) > 0:
        # Get the next node
        node = buffer.pop(0)

        # Count the number of nodes
        n_nodes += 1

        # Check if the node passes the cut
        if node.value[cut_variable_idx] < cut_value:
            # Prune the node
            node.left = None
            node.right = None
        else:
            # Add the children to the buffer
            if node.left is not None:
                buffer.append(node.left)
            if node.right is not None:
                buffer.append(node.right)

    return top_node, n_nodes

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, leaf_vs_root_factor = 0.5):

    '''
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    Based on Joel's answer at https://stackoverflow.com/a/29597209/2966723,
    but with some modifications.  

    We include this because it may be useful for plotting transmission trees,
    and there is currently no networkx equivalent (though it may be coming soon).
    
    There are two basic approaches we think of to allocate the horizontal 
    location of a node.  
    
    - Top down: we allocate horizontal space to a node.  Then its ``k`` 
      descendants split up that horizontal space equally.  This tends to result
      in overlapping nodes when some have many descendants.
    - Bottom up: we allocate horizontal space to each leaf node.  A node at a 
      higher level gets the entire space allocated to its descendant leaves.
      Based on this, leaf nodes at higher levels get the same space as leaf
      nodes very deep in the tree.  
      
    We use use both of these approaches simultaneously with ``leaf_vs_root_factor`` 
    determining how much of the horizontal space is based on the bottom up 
    or top down approaches.  ``0`` gives pure bottom up, while 1 gives pure top
    down.   
    
    
    :Arguments: 
    
    **G** the graph (must be a tree)

    **root** the root node of the tree 
    - if the tree is directed and this is not given, the root will be found and used
    - if the tree is directed and this is given, then the positions will be 
      just for the descendants of this node.
    - if the tree is undirected and not given, then a random choice will be used.

    **width** horizontal space allocated for this branch - avoids overlap with other branches

    **vert_gap** gap between levels of hierarchy

    **vert_loc** vertical location of root
    
    **leaf_vs_root_factor**

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, leftmost, width, leafdx = 0.2, vert_gap = 0.2, vert_loc = 0, 
                    xcenter = 0.5, rootpos = None, 
                    leafpos = None, parent = None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if rootpos is None:
            rootpos = {root:(xcenter,vert_loc)}
        else:
            rootpos[root] = (xcenter, vert_loc)
        if leafpos is None:
            leafpos = {}
        children = list(G.neighbors(root))
        leaf_count = 0
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            rootdx = width/len(children)
            nextx = xcenter - width/2 - rootdx/2
            for child in children:
                nextx += rootdx
                rootpos, leafpos, newleaves = _hierarchy_pos(G,child, leftmost+leaf_count*leafdx, 
                                    width=rootdx, leafdx=leafdx,
                                    vert_gap = vert_gap, vert_loc = vert_loc-vert_gap, 
                                    xcenter=nextx, rootpos=rootpos, leafpos=leafpos, parent = root)
                leaf_count += newleaves

            leftmostchild = min((x for x,y in [leafpos[child] for child in children]))
            rightmostchild = max((x for x,y in [leafpos[child] for child in children]))
            leafpos[root] = ((leftmostchild+rightmostchild)/2, vert_loc)
        else:
            leaf_count = 1
            leafpos[root]  = (leftmost, vert_loc)
#        pos[root] = (leftmost + (leaf_count-1)*dx/2., vert_loc)
#        print(leaf_count)
        return rootpos, leafpos, leaf_count

    xcenter = width/2.
    if isinstance(G, nx.DiGraph):
        leafcount = len([node for node in nx.descendants(G, root) if G.out_degree(node)==0])
    elif isinstance(G, nx.Graph):
        leafcount = len([node for node in nx.node_connected_component(G, root) if G.degree(node)==1 and node != root])
    rootpos, leafpos, leaf_count = _hierarchy_pos(G, root, 0, width, 
                                                    leafdx=width*1./leafcount, 
                                                    vert_gap=vert_gap, 
                                                    vert_loc = vert_loc, 
                                                    xcenter = xcenter)
    pos = {}
    for node in rootpos:
        pos[node] = (leaf_vs_root_factor*leafpos[node][0] + (1-leaf_vs_root_factor)*rootpos[node][0], leafpos[node][1]) 
#    pos = {node:(leaf_vs_root_factor*x1+(1-leaf_vs_root_factor)*x2, y1) for ((x1,y1), (x2,y2)) in (leafpos[node], rootpos[node]) for node in rootpos}
    xmax = max(x for x,y in pos.values())
    for node in pos:
        pos[node]= (pos[node][0]*width/xmax, pos[node][1])
    return pos

def plot_tree(graph, idx_of_feature_to_plot, label, file_name, root=None, show=False):
    if root is None:
        root = max(graph.edge_index[0]).item()
    plt.figure(1, figsize=(20, 12))
    g = to_networkx(graph, to_undirected=True)
    pos = hierarchy_pos(g, leaf_vs_root_factor=0.5, root=root)
    e = nx.draw_networkx_edges(g, pos, alpha=0.3)
    n = nx.draw_networkx_nodes(g, pos, nodelist=g.nodes(), node_color=graph.x[:,idx_of_feature_to_plot])
    cb = plt.colorbar(n)
    cb.set_label(label, size=20)
    plt.axis('off')
    plt.savefig(f"{file_name}.pdf")
    if show: plt.show()
    plt.close()