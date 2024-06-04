
from aimc_tasks.comp_graph import cgraph,core,cnodes
from tqdm import tqdm

def get_bin_of_rect(packer,rid):

    for bin in packer:
        for rect in bin:
            if rect.rid == rid:
                return bin

    # assuming packing was successful, this means the node
    # is a non-matrix node.

def get_cgraph_valid_exec_order(acc_mapping:core.Aimc_acc):
    '''
    Obtains a valid execution order for the rectpacked cgraph

    The input edge of the graph must be named x

    Returns:
    node_exec_list -- valid execution order of all cnodes, list of cnode nodes
    matrix_exec_list -- valid execution order of only matrices, list of rectpack rids
    '''

    acc_mapping.cgraph.edges['x'] = True

    # 2D list of nodes in "executable" order
    node_exec_list = []
    rid_exec_list = []
    unexecuted_nodes = acc_mapping.cgraph.nodes

    def set_node_outputs_to_true(node : cnodes.Node):
        for edgename in node.outputs:
            acc_mapping.cgraph.edges[edgename] = True
        return

    totalnodes = len(acc_mapping.cgraph.nodes)
    with tqdm(total=totalnodes,desc='Executed Nodes') as pbar:
        while unexecuted_nodes != []:
            this_active_set = []

            # look through unexecuted nodes to see which ones are active
            for node in unexecuted_nodes:
                if(acc_mapping.cgraph.check_if_node_ready(node)):
                    this_active_set.append( node )

            # only execute one node from each bin
            active_bins = []
            for node in this_active_set:
                # check if it's a packable node in the first place
                if hasattr(node,'matrix'):
                    bin = get_bin_of_rect(acc_mapping.packer,node.rid)
                    if bin in active_bins:
                        this_active_set.remove( node )
                    else:
                        active_bins.append(bin)

            # execute
            executed_matrix_node_rids = []
            for node in this_active_set:
                unexecuted_nodes.remove(node)
                set_node_outputs_to_true(node)
                if hasattr(node,'matrix'):
                    executed_matrix_node_rids.append(node.rid)

            node_exec_list.append([nd.rid for nd in this_active_set])

            if executed_matrix_node_rids != []:
                rid_exec_list.append(executed_matrix_node_rids)

            pbar.update(totalnodes - len(unexecuted_nodes))

    return node_exec_list, rid_exec_list

def get_nsim_SA_assignments(acc_mapping:core.Aimc_acc):

    visited = []

    for node in acc_mapping.cgraph.nodes:
        for neighbor_node in acc_mapping.get_neighbors(node):
            if neighbor_node not in visited:
                visited.append(neighbor_node)
                acc_mapping(neighbor_node)      

    return