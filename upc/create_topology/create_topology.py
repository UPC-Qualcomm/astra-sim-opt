"""
Topology classes for HPC Interconnects
Supports: Dragonfly, Folded-Clos, Jellyfish
"""

import json, sys
import time,pprint
import random
import networkx as nx
from itertools import permutations, product

class CustomizedDragonfly:
    def __init__(self, G, A, h=1, concentration=1, bandwidth_config=None):
        self.name = "dragonfly"
        self.num_groups = G # number of groups
        self.num_switches = A # number of switches per group
        self.num_interpod_links_per_switch = h # number of interpod links per switch
        self.concentration_factor = concentration # number of host servers per switch
        self.total_num_switches = G * A # total number of switches in the topology
        self.numHostsPerPod = self.num_switches * self.concentration_factor
        self.total_num_hosts = self.numHostsPerPod * self.num_groups
        self.adjacency_matrix = None
        self.interpod_links = []
        self.intrapod_links = []
        
        # Bandwidth Configuration
        # Expected keys: 'host_switch', 'intra_group', 'inter_group'
        self.bandwidth_config = bandwidth_config if bandwidth_config else {}
        self.link_bandwidths = {} 
   
    def NumServers(self):
        return self.total_num_switches * self.concentration_factor

    def NumLinks(self):
        return 2 * int(self.num_groups*self.num_switches*(self.num_switches-1)/2 +
            self.num_groups*self.num_switches*int(float(self.num_interpod_links_per_switch))/2) # total number of directed links

    def NumSwitches(self):
        return self.total_num_switches

    def DesignIntraGroupTopology(self):
        self.adjacency_matrix = [0] * self.total_num_switches
        for switch in range(self.total_num_switches):
            self.adjacency_matrix[switch] = [0] * self.total_num_switches
        #first design the intragroup matrix in the full topology
        for i in range(self.num_groups):
            for row in range(i * self.num_switches, (i+1) * self.num_switches):
                for col in range(row+1, (i+1) * self.num_switches):
                    if row != col:
                        self.adjacency_matrix[row][col] = 1
                        self.adjacency_matrix[col][row] = 1
                        self.intrapod_links.append((row,col))

    # Non-canonical Dragonfly with potentially non-even distribution of links without any randomness
    # note: if num_intergroup_links_per_group is odd, then there will be a group with num_intergroup_links_per_group-1 interpod links
    def DesignGroupLevelTopology(self):
        num_intergroup_links_per_group = self.num_switches * self.num_interpod_links_per_switch
        eta = [[0]*self.num_groups for _ in range(self.num_groups)]
        for i in range(self.num_groups):
            for d in range(num_intergroup_links_per_group):
                mu = [num_intergroup_links_per_group+1] * self.num_groups
                for j in range(self.num_groups):
                    if i == j: continue
                    mu[j] = sum(eta[j])
                k = mu.index(min(mu))
                if i != k and sum(eta[k]) < num_intergroup_links_per_group and sum(eta[i]) < num_intergroup_links_per_group:
                    eta[i][k] += 1
                    eta[k][i] += 1
        return eta

    def DesignFullTopology(self):
        self.DesignIntraGroupTopology()
        eta = self.DesignGroupLevelTopology()
        for i in range(len(eta)):
            for j in range(i+1, len(eta[i])):
                while eta[i][j] != 0:
                    src, dst = self.find_available_src_dst(i,j)
                    eta[i][j] -= 1
                    self.adjacency_matrix[src][dst] += 1
                    self.adjacency_matrix[dst][src] += 1
                    self.interpod_links.append((src,dst))
        return

    def find_available_src_dst(self, i, j):
        src_group = i
        src_group_switches = range(src_group * self.num_switches, (src_group+1) * self.num_switches)
        src_mu = []
        for src_switch in src_group_switches:
            src_mu.append(sum(self.adjacency_matrix[src_switch]))
        src = src_mu.index(min(src_mu)) + (src_group) * self.num_switches
       
        dst_group = j
        dst_group_switches = range(dst_group * self.num_switches, (dst_group+1) * self.num_switches)
        dst_mu = []
        for dst_switch in dst_group_switches:
            dst_mu.append(sum(self.adjacency_matrix[dst_switch]))
        dst = dst_mu.index(min(dst_mu)) + (dst_group) * self.num_switches
        return src, dst

    # Exactly as described in the paper (with some randomness)
    def DesignGroupLevelTopologyV2(self):
        self.DesignIntraGroupTopology()
        #first distribute the interpod links on a pod level
        num_intergroup_links_per_group = self.num_switches * self.concentration_factor
        eta = [[0]*self.num_groups for _ in range(self.num_groups)]
        for k in range(self.num_groups):
            for d in range(num_intergroup_links_per_group):
                mu = [num_intergroup_links_per_group+1] * self.num_groups
                for i in range(self.num_groups):
                    if i == k: continue
                    mu[i] = 2*eta[i][k] + sum([eta[i][j] for j in range(self.num_groups) if j!= k])
                i_p = mu.index(min(mu))
                if sum(eta[i_p]) < num_intergroup_links_per_group and i_p != k:
                    eta[i_p][k] += 1
        return

    def writeTopologyToFile(self, path):
        with open(path, "w") as fp:
            json.dump(self.adjacency_matrix, fp)
   
    def loadTopologyFromFile(self, path):
        with open(path, "r") as fp:
            self.adjacency_matrix = json.load(fp)

    def writeLinksToFile(self, path):
        link_dict = {"interpod_links":self.interpod_links, "intrapod_links":self.intrapod_links}
        with open(path, 'w') as fp:
            json.dump(link_dict, fp)
   
    def loadLinksFromFile(self, path):
        link_dict = {}
        with open(path, "r") as fp:
            link_dict = json.load(fp)
        self.interpod_links = link_dict['interpod_links']
        self.intrapod_links = link_dict['intrapod_links']

    def WriteAdjacencyMatrixToG2ConfFile(self):
        links = []
        str_builder = "links: "
        count = 1
        for src in range(self.total_num_switches):
            for dst in range(src, self.total_num_switches):
                if self.adjacency_matrix[src][dst] != 0:
                    str_builder += ("({},s{},s{});".format(count,src+1, dst+1))
                    links.append( ("s{}".format(src+1), "s{}".format(dst+1) ) )
                    count += 1
        host_num = 1
        for switch_num in range(1, self.total_num_switches + 1):
            for _ in range(self.concentration_factor):
                str_builder += ("({},h{},s{});".format(count, host_num, switch_num))
                links.append( ("h{}".format(host_num), "s{}".format(switch_num)) )
                host_num += 1
                count += 1
        return links, str_builder, count
    
    def LinksToG2ConfFile(self):
        # Ensure self.links is populated with switch-to-switch links
        if not hasattr(self, 'links') or not self.links:
            self.writeAdjacencyMatrixToLinks()

        # Start with a copy of the existing switch-to-switch links dictionary
        links = self.links.copy()
        
        # Continue numbering from the last link ID
        link_id = max(links.keys()) + 1
        
        # Add host-to-switch links
        host_bw = self.bandwidth_config.get('host_switch', 1.0)
        host_num = 1
        for switch_num in range(1, self.total_num_switches + 1):
            for _ in range(self.concentration_factor):
                # Add bidirectional links for hosts
                links[link_id] = (f"h{host_num}", f"s{switch_num}")
                self.link_bandwidths[link_id] = host_bw
                link_id += 1
                
                links[link_id] = (f"s{switch_num}", f"h{host_num}")
                self.link_bandwidths[link_id] = host_bw
                link_id += 1
                
                host_num += 1
        return links
   
    def writeAdjacencyMatrixToLinks(self):
        self.links = {}
        # self.link_bandwidths is initialized in __init__ but we populate switch links here
        # Note: This clears previous switch links but keeps the dict object
        
        intra_bw = self.bandwidth_config.get('intra_group', 1.0)
        inter_bw = self.bandwidth_config.get('inter_group', 1.0)
        
        # Convert interpod_links to set for faster lookup
        interpod_set = set(self.interpod_links)

        link_id = 1
        for src in range(self.total_num_switches):
            for dst in range(src, self.total_num_switches):
                if self.adjacency_matrix[src][dst] != 0:
                    self.links[link_id] = ("s{}".format(src+1), "s{}".format(dst+1))
                    
                    # Determine bandwidth
                    if (src, dst) in interpod_set or (dst, src) in interpod_set:
                        self.link_bandwidths[link_id] = inter_bw
                    else:
                        self.link_bandwidths[link_id] = intra_bw
                        
                    link_id += 1
        
        # Add reverse links
        for i in range(link_id - 1):
            self.links[i+link_id] = (self.links[i + 1][1], self.links[i + 1][0])
            self.link_bandwidths[i+link_id] = self.link_bandwidths[i+1]
   
    def addFlowsToFlowDict(self,flow_id,flowLinks):
        self.F[str(flow_id)] = []
        for (x,y) in flowLinks:
            if x+'-'+y in self.reverseL:
                self.F[str(flow_id)].append(self.reverseL[x+'-'+y])
            elif y+'-'+x in self.reverseL:
                self.F[str(flow_id)].append(self.reverseL[y+'-'+x])
            else:
                self.F[str(flow_id)] = []

    def addFlows(self, i, j): # for ECMP routing
        src = "s"+str(i+1)
        dst = "s"+str(j+1)
        path_lists = self.paths[src][dst]
        path_list_len = len(path_lists)
        for path_list in path_lists:
            flow_links = [(x,y) for x,y in zip(path_list, path_list[1:])]
            self.addFlowsToFlowDict(self.flow_id,flow_links)
            self.sorted_traffic[self.flow_id] = self.tm[i][j] // path_list_len
            self.flow_id += 1
   
    def GenerateUniformRouting(self):
        self.writeAdjacencyMatrixToLinks()
        G = nx.DiGraph(self.links.values())
        for link in self.interpod_links:
            G.add_edge("s{}".format(link[0]+1),"s{}".format(link[1]+1),weight=2)
            G.add_edge("s{}".format(link[1]+1),"s{}".format(link[0]+1),weight=2)
        paths = nx.shortest_path(G, weight='weight')
        return paths

    def GenerateShortestPathFlowDict(self, tm):
        print("*** Generating shortest path flow dict for dragonfly... ")
        paths = self.GenerateUniformRouting()
        self.F = {}
        self.sorted_traffic = {}
        flow_id = 1
        self.reverseL = dict( (v[0]+"-"+v[1],k)for k,v in self.links.items() )
        for i in range(len(tm)):
            for j in range(i+1, len(tm[i])):
                src = "s"+str(i+1)
                dst = "s"+str(j+1)
                if tm[i][j] != 0:
                    forwardPathList = paths[src][dst]
                    forwardFlowLinks = [(x,y) for x,y in zip(forwardPathList, forwardPathList[1:])]
                    self.addFlowsToFlowDict(flow_id,forwardFlowLinks)
                    self.sorted_traffic[flow_id] = tm[i][j]
                    flow_id += 1
                if tm[j][i] != 0:
                    backwardPathList = paths[dst][src]
                    backwardPathList = [(x,y) for x,y in zip(backwardPathList, backwardPathList[1:])]
                    self.addFlowsToFlowDict(flow_id,backwardPathList)
                    self.sorted_traffic[flow_id] = tm[j][i]
                    flow_id += 1
        assert(self.F),"Error: empty return flow dict"
        return self.F, self.sorted_traffic

    def GenerateECMPFlowDict(self, tm):
        print("*** Generating ECMP flow dict for dragonfly... ")
        self.writeAdjacencyMatrixToLinks()
        G = nx.DiGraph(self.links.values())
        # for link in self.interpod_links:
        #     G.add_edge("s{}".format(link[0]+1),"s{}".format(link[1]+1))
        self.paths = {} # paths = defaultdict(lambda: defaultdict(list))
        # count = 0
        # all-to-all routing paths
        for i in range(1, int(self.total_num_switches+1)):
            src = "s{}".format(i)
            self.paths[src] = {}
            for j in range(1, int(self.total_num_switches+1)):
                if i != j:
                    dst = "s{}".format(j)
                    self.paths[src][dst] = list(nx.all_shortest_paths(G,source=src,target=dst, weight="weight"))
                    # if len(self.paths[src][dst]) > 1: count += 1
       
        self.tm = tm
        self.F = {}
        self.sorted_traffic = {}
        self.reverseL = dict( (v[0]+"-"+v[1],k)for k,v in self.links.items() )
        self.flow_id = 1
        for i in range(len(tm)):
            for j in range(i+1, len(tm[i])):
                if tm[i][j] != 0:
                    self.addFlows(i, j)
                if tm[j][i] != 0:
                    self.addFlows(j, i)
        return self.F, self.sorted_traffic


class Jellyfish():
    def __init__(self, num_switches, degree, seed = 0, num_hosts_per_switch=1, bandwidth_config=None):
        self.name = "jellyfish"
        self.num_switches = num_switches
        self.degree = degree
        self.num_hosts_per_switch = num_hosts_per_switch
        self.num_intrapod_links_per_switch = self.degree
        self.seed = seed
        random.seed(self.seed)
        self.adjacency_matrix = None
        self.links = None
        self.total_num_hosts = self.num_switches * self.num_hosts_per_switch
        self.nodes = ['s'+str(i) for i in range(1, num_switches+1)]
        
        # Bandwidth Configuration
        # Expected keys: 'host_switch', 'switch_switch'
        self.bandwidth_config = bandwidth_config if bandwidth_config else {}
        self.link_bandwidths = {}

    def NumServers(self):
        return int(self.num_switches*self.num_hosts_per_switch)

    def NumLinks(self):
        assert(self.adjacency_matrix), "Need to construct full topology first!"
        self.num_links = 0 # undirected links (bidirectional flow)
        for src in range(len(self.adjacency_matrix)):
            for dst in range(src+1, len(self.adjacency_matrix[src])):
                self.num_links += 2 * self.adjacency_matrix[src][dst]
        return self.num_links

    def NumSwitches(self):
        return self.num_switches

    def makeLinksFromAdjMatrix(self):
        # assume one host per switch for now
        # assume undirected links --> bidirectional flows
        assert(self.adjacency_matrix), "Need to construct full topology first!"
        self.links = {}
        self.link_weight = {}
        
        switch_bw = self.bandwidth_config.get('switch_switch', 1.0)
        
        linkID = 1
        for src in range(len(self.adjacency_matrix)):
            for dst in range(src+1, len(self.adjacency_matrix[src])):
                link_count = self.adjacency_matrix[src][dst]
                while link_count > 0:
                    self.links[linkID] = ("s" + str(src+1), "s" + str(dst+1))
                    self.link_bandwidths[linkID] = switch_bw
                    linkID += 1
                    
                    self.links[linkID] = ("s" + str(dst+1), "s" + str(src+1))
                    self.link_bandwidths[linkID] = switch_bw
                    linkID += 1
                    
                    link_count -= 1
        assert(linkID == self.NumLinks() + 1)

    def deletePairFromVector(self,pair, vector):
        if pair in vector:
            orig_size = len(vector)
            vector.remove(pair)
            assert(orig_size - 1 == len(vector))

    def DesignFullTopology(self):
#        print("*** Constructing Jellyfish Topology...")
        self.adjacency_matrix = [[0]*self.num_switches for _ in range(self.num_switches)]
        formed_links = [0] * self.num_switches
        # first, form a basic full mesh if there are more links per switch than there are switches
        full_mesh_n = 0
        num_intrapod_links_per_switch_copy = self.num_intrapod_links_per_switch
        while num_intrapod_links_per_switch_copy >= self.num_switches - 1:
            full_mesh_n += 1
            formed_links = [full_mesh_n * (self.num_switches - 1)] * self.num_switches
            num_intrapod_links_per_switch_copy -= self.num_switches - 1
        for i in range(self.num_switches):
            self.adjacency_matrix[i] = [full_mesh_n] * self.num_switches
            self.adjacency_matrix[i][i] = 0
        # now form the remainder of the connections
        link_pairs = []
        for src in range(self.num_switches):
            potential_targets = list(range(self.num_switches))
            random.shuffle(potential_targets)
            offset = 0
            while formed_links[src] < self.num_intrapod_links_per_switch and offset < self.num_switches:
                dst = potential_targets[offset]
                if formed_links[dst] < self.num_intrapod_links_per_switch and dst != src:
                    self.adjacency_matrix[src][dst] += 1
                    self.adjacency_matrix[dst][src] += 1
                    formed_links[src] += 1
                    formed_links[dst] += 1
                    link_pairs.append((min(src, dst), max(src , dst)))
                offset += 1
        random.shuffle(link_pairs)
        for i in range(self.num_switches):
            while self.num_intrapod_links_per_switch - formed_links[i] >= 2:
                for pair_id in range(len(link_pairs)):
                    found = False
                    if link_pairs[pair_id][0] != i and link_pairs[pair_id][1] != i:
                        sw1 = link_pairs[pair_id][0]
                        sw2 = link_pairs[pair_id][1]
                        self.adjacency_matrix[sw1][sw2] -= 1
                        self.adjacency_matrix[sw2][sw1] -= 1
                        self.adjacency_matrix[i][sw1] += 1
                        self.adjacency_matrix[sw1][i] += 1
                        self.adjacency_matrix[i][sw2] += 1
                        self.adjacency_matrix[sw2][i] += 1
                        formed_links[i] += 2
                        link_pairs.pop(pair_id)
                        found = True
                    if found:
                        break
        self.makeLinksFromAdjMatrix()

    def WriteAdjacencyMatrixToG2ConfFile(self):
        
        str_builder = "links: "
        count = 1
        #for link_id, (src, dst) in self.links.items():
        #    str_builder += ("({},s{},s{});".format(count,src+1, dst+1))

        return self.links, str_builder, count

    def LinksToG2ConfFile(self):
        # Ensure self.links is populated with switch-to-switch links
        if not self.links:
            self.makeLinksFromAdjMatrix()

        # Start with a copy of the existing switch-to-switch links dictionary
        links = self.links.copy()

        # Continue numbering from the last link ID
        link_id = max(links.keys()) + 1

        # Add host-to-switch links
        host_bw = self.bandwidth_config.get('host_switch', 1.0)
        
        host_num = 1
        for switch_num in range(1, self.num_switches + 1):
            for _ in range(self.num_hosts_per_switch):
                # Add bidirectional links for hosts
                links[link_id] = (f"h{host_num}", f"s{switch_num}")
                self.link_bandwidths[link_id] = host_bw
                link_id += 1
                
                links[link_id] = (f"s{switch_num}", f"h{host_num}")
                self.link_bandwidths[link_id] = host_bw
                link_id += 1
                
                host_num += 1
        return links
    
    def addFlowsToFlowDict(self,flow_id,flowLinks):
        self.F[flow_id] = []
        for (x,y) in flowLinks:
            if x+'-'+y in self.reverseL:
                self.F[flow_id].append(self.reverseL[x+'-'+y])
            elif y+'-'+x in self.reverseL:
                self.F[flow_id].append(self.reverseL[y+'-'+x])
            else:
                self.F[flow_id] = []

    def addFlows(self, i, j): # for ECMP routing
        src = "s"+str(i+1)
        dst = "s"+str(j+1)
        path_lists = self.paths[src][dst]
        path_list_len = len(path_lists)
        for path_list in path_lists:
            flow_links = [(x,y) for x,y in zip(path_list, path_list[1:])]
            self.addFlowsToFlowDict(self.flow_id,flow_links)
            self.sorted_traffic[self.flow_id] = self.tm[i][j] // path_list_len
            self.flow_id += 1

    def GenerateUniformRouting(self):
        self.makeLinksFromAdjMatrix()
        G = nx.MultiDiGraph()
        G.add_edges_from(self.links.values())
        self.paths = (nx.shortest_path(G))
        return self.paths

    def GenerateShortestPathFlowDict(self, tm):
        print("*** Generating flow dict for Jellyfish...")
        self.GenerateUniformRouting()
        self.F = {}
        self.sorted_traffic = {}
        flow_id = 1
        self.reverseL = dict( (v[0]+"-"+v[1],k)for k,v in self.links.items() )
        for i in range(len(tm)):
            for j in range(i+1, len(tm[i])):
                src = "s" + str(i+1)
                dst = "s" + str(j+1)
                if tm[i][j] != 0:
                    forwardPathList = self.paths[src][dst]
                    forwardFlowLinks = [(x,y) for x,y in zip(forwardPathList, forwardPathList[1:])]
                    self.addFlowsToFlowDict(flow_id,forwardFlowLinks)
                    self.sorted_traffic[flow_id] = tm[i][j]
                    flow_id += 1
                if tm[j][i] != 0:
                    backwardPathList = self.paths[dst][src]
                    backwardFlowLinks = [(x,y) for x,y in zip(backwardPathList, backwardPathList[1:])]
                    self.addFlowsToFlowDict(flow_id,backwardFlowLinks)
                    self.sorted_traffic[flow_id] = tm[j][i]
                    flow_id += 1
        assert(self.F),"Error: empty return flow dict"
        return self.F, self.sorted_traffic

   
    def GenerateECMPFlowDict(self, tm):
        print("*** Generating flow dict for Jellyfish...")
        self.makeLinksFromAdjMatrix()
        G = nx.MultiDiGraph()
        G.add_edges_from(self.links.values())
        # paths = (nx.shortest_path(G))
        self.paths = {}
        for i in range(1, int(self.num_switches+1)):
            src = "s{}".format(i)
            self.paths[src] = {}
            for j in range(1, int(self.num_switches+1)):
                if i != j:
                    dst = "s{}".format(j)
                    self.paths[src][dst] = list(nx.all_shortest_paths(G,source=src,target=dst))
        self.tm = tm
        self.F = {}
        self.sorted_traffic = {}
        self.reverseL = dict( (v[0]+"-"+v[1],k)for k,v in self.links.items() )
        self.flow_id = 1
        for i in range(len(tm)):
            for j in range(i+1, len(tm[i])):
                if tm[i][j] != 0:
                    self.addFlows(i, j)
                if tm[j][i] != 0:
                    self.addFlows(j, i)
        return self.F, self.sorted_traffic


class FoldedClos():
    def __init__(self, K, link_capacity=1, N=3, bandwidth_config=None):
        self.name = "folded_clos"
        self.K = K
        self.numCoreSwitches = K**2 / 4
        self.numHosts = K**3 / 4
        self.numSwitchesPerPod = K
        self.numHostsPerPod = K**2 / 4
        self.numSwitchPorts = K
        self.totalNumSwitches = int(self.numSwitchesPerPod * self.K + self.numCoreSwitches)
        self.total_num_hosts = self.numHosts
        self.links = {}
        self.c_dict = None
        self.link_capacity = link_capacity
        self.adjacency_matrix = [[0] * self.totalNumSwitches for _ in range(self.totalNumSwitches)]
        self.hosts = ['h'+str(i) for i in range(1,int(self.numHosts+1))]
        self.num_switches = sum([K**n for n in range(N)])
        self.switches = ['s'+str(i) for i in range(1, self.totalNumSwitches+1)]
        self.nodes = self.hosts + self.switches
        
        # Bandwidth Configuration
        # Expected keys: 'host_edge', 'edge_agg', 'agg_core'
        self.bandwidth_config = bandwidth_config if bandwidth_config else {}
        self.link_bandwidths = {}

    def NumServers(self):
        return int(self.numHosts)

    def NumLinks(self):
        return int(len(self.links))

    def NumSwitches(self):
        return self.totalNumSwitches

    def DesignFullTopology(self):
        # links = {}
        linkID = 1
        
        bw_host_edge = self.bandwidth_config.get('host_edge', 1.0)
        bw_edge_agg = self.bandwidth_config.get('edge_agg', 1.0)
        bw_agg_core = self.bandwidth_config.get('agg_core', 1.0)
        
        for pod in range(self.K):
            hostIDStart = int(pod * self.numHostsPerPod + 1)
            coreSwitchStart = int(self.numSwitchesPerPod * self.K + 1)
            EdgeSwitchStart =  int(pod*self.numSwitchesPerPod + 1)
            EdgeSwitchEnd = int((pod + 1/2) * self.numSwitchesPerPod)
            aggSwitchStart = int(EdgeSwitchEnd + 1)
            aggSwitchEnd = int(EdgeSwitchEnd + self.numSwitchesPerPod // 2)

            for eS in range(EdgeSwitchStart,EdgeSwitchEnd + 1):
                for h in range(self.numSwitchPorts // 2):        
                    self.links[linkID] = ('h'+str(hostIDStart + h), 's'+str(eS))
                    self.link_bandwidths[linkID] = bw_host_edge
                    linkID += 1
                hostIDStart += self.numSwitchPorts // 2
            for aS in range(aggSwitchStart,aggSwitchEnd + 1):
                for eS in range(EdgeSwitchStart,EdgeSwitchEnd + 1):
                    self.links[linkID] = ('s'+str(aS), 's'+str(eS))
                    self.link_bandwidths[linkID] = bw_edge_agg
                    # self.adjacency_matrix[aS][eS] += 1
                    # self.adjacency_matrix[eS][aS] += 1
                    linkID += 1
                for cS in range(self.numSwitchPorts//2):
                    self.links[linkID] = ('s'+str(coreSwitchStart + cS), 's'+str(aS))
                    self.link_bandwidths[linkID] = bw_agg_core
                    # self.adjacency_matrix[coreSwitchStart + cS][aS] += 1
                    # self.adjacency_matrix[aS][coreSwitchStart + cS] += 1
                    linkID += 1
                coreSwitchStart += self.numSwitchPorts//2
        # links are directed
        for i in range(linkID - 1):
            self.links[i + linkID] = (self.links[i + 1][1], self.links[i + 1][0])
            self.link_bandwidths[i + linkID] = self.link_bandwidths[i + 1]
            
        self.num_links = linkID
        # C = {i:args.c for i in range(1, 2 * linkID - 1)}
        # self.c_dict = {i:self.link_capacity for i in range(1, 2 * linkID - 1)}

    def LinksToG2ConfFile(self):
        str_builder = "links: "
        for id, link in self.links.items():
            str_builder += ("(s{},s{});".format(link[0], link[1]))
        return self.links, str_builder

    def GenerateUniformRouting(self, tapering_num = 0):
        """Generate the shortest path routing with balanced link loads for the clos-pods structure.
        The routing algorithm is compatiable with the algorithm described in http://ccr.sigcomm.org/online/files/p63-alfares.pdf
        Args:
            links (dict): Dictionary of the link information (key, value) = (linkID, (src, dst)).
            K (int): The number of pods in the clos-pods structure.
        Returns:
            list: The routing list Paths[src][dst] = [all transversed nodes].
        Examples:
            paths = GenerateUniformRouting(links, 4)
        """

        assert (self.numCoreSwitches.is_integer() and self.numHosts.is_integer()),"K is not appropriate!"
        # start_time = time.time()
        G = nx.Graph()
        G.add_edges_from(self.links.values())
        # print("Time to build networks:", str(time.time() - start_time))
        paths = (nx.shortest_path(G))
        # print("Time to find all shortest paths:", str(time.time() - start_time))
        # Intra-Pod inter-EdgeSwitch routing
        # If source is the ith host in mth edge switch and destination is the jth host in nth edge switch,
        # the routing will transverse ath aggregation switch, a = (j + m) mod (k/2).
        # The remaining transversed nodes are then deterministic.
        for podID in range(self.K):
            for esrc, edst in permutations(range(1, self.numSwitchesPerPod // 2 + 1),2):
                for hsrc, hdst in product(range(1, self.numSwitchPorts // 2+1),range(1, self.numSwitchPorts // 2 + 1)):
                    eOutPort = (hdst - 1 + esrc - 1) % (self.numSwitchesPerPod // 2) + 1
                    hSrcID = podID * self.numHostsPerPod + (esrc - 1)*self.numSwitchPorts // 2 + hsrc
                    hDstID = podID * self.numHostsPerPod + (edst - 1)*self.numSwitchPorts // 2 + hdst
                    eSrcID = podID * self.numSwitchesPerPod + esrc
                    eDstID = podID * self.numSwitchesPerPod + edst
                    aID    = podID * self.numSwitchesPerPod + self.numSwitchesPerPod // 2 + eOutPort
                    paths["h" + str(int(hSrcID))]["h" + str(int(hDstID))] = ['h' + str(int(hSrcID)),'s' + str(int(eSrcID)),'s' + str(int(aID)),'s' + str(int(eDstID)),'h' + str(int(hDstID))]
        # Inter-Pod routing
        # If source is the ith host in mth edge switch within pod k1, and destination is the jth host in nth edge switch within pod k2,
        # the routing will transverse ath aggregation switch in pod k1, a = (j + m) mod (k/2),
        # and transverse the kth core switch, k = (a + m) mod (k/2).
        # The remaining transversed nodes are then deterministic.

        for psrc, pdst in permutations(range(self.K),2):
            for esrc, edst in product(range(1, self.numSwitchesPerPod // 2 + 1), range(1, self.numSwitchesPerPod // 2 + 1)):
                for hsrc, hdst in product(range(1, self.numSwitchPorts // 2 + 1),range(1, self.numSwitchPorts // 2 + 1)):
                    eOutPort = (hdst - 1 + esrc -1) % (self.numSwitchesPerPod // 2) + 1
                    hSrcID = int(psrc * self.numHostsPerPod + (esrc - 1) * self.numSwitchPorts // 2 + hsrc)
                    hDstID = int(pdst * self.numHostsPerPod + (edst - 1)*self.numSwitchPorts // 2 + hdst)
                    eSrcID = int(psrc * self.numSwitchesPerPod + esrc)
                    eDstID = int(pdst * self.numSwitchesPerPod + edst)
                    aSrcID = int(psrc * self.numSwitchesPerPod + self.numSwitchesPerPod // 2 + eOutPort)
                    aDstID = int(pdst * self.numSwitchesPerPod + self.numSwitchesPerPod // 2 + eOutPort)
                    aOutPort = int((hdst - 1 + eOutPort - 1) % (self.numSwitchesPerPod // 2 - tapering_num) + 1)
                    coreID = int(self.numSwitchesPerPod * self.K + (eOutPort - 1) * self.numSwitchPorts // 2 + aOutPort)

                    paths["h" + str(hSrcID)]["h" + str(hDstID)] = ['h' + str(hSrcID),'s' + str(eSrcID),'s' + str(aSrcID),'s' + str(coreID),'s' + str(aDstID),'s' + str(eDstID),'h' + str(hDstID)]        
        # print("Time to generate all clos paths:", str(time.time() - start_time))
        return paths

    def addFlowsToFlowDict(self,flow_id,flowLinks):
        self.F[flow_id] = []
        for (x,y) in flowLinks:
            if x+'-'+y in self.reverseL:
                self.F[flow_id].append(self.reverseL[x+'-'+y])
            elif y+'-'+x in self.reverseL:
                self.F[flow_id].append(self.reverseL[y+'-'+x])
            else:
                self.F[flow_id] = []

    def addFlows(self, i, j): # for ECMP routing
        src = "h"+str(i+1)
        dst = "h"+str(j+1)
        path_lists = self.paths[src][dst]
        path_list_len = len(path_lists)
        for path_list in path_lists:
            flow_links = [(x,y) for x,y in zip(path_list, path_list[1:])]
            self.addFlowsToFlowDict(self.flow_id,flow_links)
            self.sorted_traffic[self.flow_id] = self.tm[i][j] // path_list_len
            self.flow_id += 1

    def GenerateShortestPathFlowDict(self, tm):
        print("*** Generating flow dict for folded clos K = {}".format(self.K))
        self.paths = self.GenerateUniformRouting()
        # start_time = time.time()
        self.F = {}
        self.sorted_traffic = {}
        flow_id = 1
        self.reverseL = dict( (v[0]+"-"+v[1],k)for k,v in self.links.items() )
        for i in range(len(tm)):
            for j in range(i+1, len(tm[i])):
                src = "h"+str(i+1)
                dst = "h"+str(j+1)
                if tm[i][j] != 0:
                    forwardPathList = self.paths[src][dst]
                    forwardFlowLinks = [(x,y) for x,y in zip(forwardPathList, forwardPathList[1:])]
                    self.addFlowsToFlowDict(flow_id,forwardFlowLinks)
                    self.sorted_traffic[flow_id] = tm[i][j]
                    flow_id += 1
                if tm[j][i] != 0:
                    backwardPathList = self.paths[dst][src]
                    backwardFlowLinks = [(x,y) for x,y in zip(backwardPathList, backwardPathList[1:])]
                    self.addFlowsToFlowDict(flow_id,backwardFlowLinks)
                    self.sorted_traffic[flow_id] = tm[j][i]
                    flow_id += 1
        assert(self.F),"Error: empty return flow dict"
        # print("Time to generate f dict:", str(time.time() - start_time))
        return self.F, self.sorted_traffic

    def GenerateECMPFlowDict(self, tm):
        print("*** Generating ECMP flow dict for folded clos K = {}".format(self.K))
        G = nx.DiGraph(self.links.values())
        self.paths = {}
        for i in range(1, int(self.numHosts+1)):
            src = "h{}".format(i)
            self.paths[src] = {}
            for j in range(1, int(self.numHosts+1)):
                if i != j:
                    dst = "h{}".format(j)
                    self.paths[src][dst] = list(nx.all_shortest_paths(G,source=src,target=dst, weight="weight"))
        self.tm = tm
        self.F = {}
        self.sorted_traffic = {}
        self.reverseL = dict( (v[0]+"-"+v[1],k)for k,v in self.links.items() )
        self.flow_id = 1
        for i in range(len(tm)):
            for j in range(i+1, len(tm[i])):
                if tm[i][j] != 0:
                    self.addFlows(i, j)
                if tm[j][i] != 0:
                    self.addFlows(j, i)
        return self.F, self.sorted_traffic