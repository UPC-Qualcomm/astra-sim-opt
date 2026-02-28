"""
Approximated topology generators for FoldedClos, Dragonfly, and Jellyfish.

Core idea
---------
Everything from the ToR (edge switch) downward — hosts (h*), NVSwitches (v*),
NIC/NPU-switches (n*, p*) and the ToR switches themselves (t*) — is kept
UNCHANGED.  The "upper fabric" is replaced with a compact bandwidth-preserving
tree:

  FoldedClos : per-pod agg group  →  one 'g{pod}' switch per pod
               all core switches  →  one 'r0' root switch

  Dragonfly  : intra-group mesh   →  one 'g{group}' group aggregator per group
               all inter-group    →  one 'r0' root switch

  Jellyfish  : random switch mesh →  one 'r0' root switch
               (all switches ARE ToRs, so only 2-level star above them)

In every case the link BW on the new upper-fabric link equals the SUM of
all original link BWs it replaces, preserving total available bandwidth.
Because the resulting upper fabric is a tree, routing is deterministic and
unique — no ECMP choices needed.
"""

import re
import networkx as nx
from collections import defaultdict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_num(name):
    m = re.search(r'\d+', name)
    return int(m.group()) if m else -1


def _add_bidir(new_links, new_bw, new_lat, ctr, src, dst, bw, lat):
    """Append a bidirectional link pair and return the incremented counter."""
    new_links[ctr] = (src, dst);  new_bw[ctr] = bw;  new_lat[ctr] = lat;  ctr += 1
    new_links[ctr] = (dst, src);  new_bw[ctr] = bw;  new_lat[ctr] = lat;  ctr += 1
    return ctr


# ---------------------------------------------------------------------------
# FoldedClos approximation
# ---------------------------------------------------------------------------

def approximate_foldedclos(topo_obj):
    """
    Approximate a FoldedClos topology.

    Kept unchanged
    --------------
    - NPU nodes (h*), NVSwitch nodes (v*), NIC/NPU-switch nodes (n*, p*)
    - Edge / ToR switches (t*)
    - All links that involve only the above node types

    Replaced
    --------
    - K/2 agg switches per pod  →  one 'g{i}' switch per pod  (i = 0, 1, ...)
    - All (K/2)^2 core switches →  one 'r0' root switch

    Bandwidth aggregation
    ---------------------
    Bidirectional links appear as two directed edges in the link dict.
    Only the forward direction is counted so each physical link is
    counted exactly once.

    edge_sw → g{pod}  :  sum over each agg switch a in the pod of
                         BW(edge_sw → a)   [one direction only]
    g{pod}  → r0      :  sum over each agg switch a in the pod of
                         BW(a → core_sw)   [one direction only, summed
                         over all core switches a connects to]

    Args
    ----
    topo_obj : FoldedClos instance after DesignFullTopology() + LinksToG2ConfFile()

    Returns
    -------
    links (dict), link_bandwidths (dict), link_latencies (dict),
    metadata (dict):
        metadata['tor_switches']   – frozenset of ToR switch names
        metadata['upper_switches'] – list of new upper-fabric switch names
        metadata['tor_to_group']   – dict  tor_name → group_switch_name
        metadata['group_to_core']  – dict  group_switch_name → 'r0'
    """
    edge_set = set(topo_obj.edge_switches)
    agg_set  = set(topo_obj.agg_switches)
    core_set = set(topo_obj.core_switches)

    # ── Scan original links to accumulate per-(edge,agg) and per-agg BW ──
    # Bidirectional links are stored as two directed edges in topo_obj.links.
    # We only count the forward direction (edge→agg, agg→core) so each
    # physical link is counted exactly once per (edge_sw, agg_sw) pair.
    edge_agg_bw  = defaultdict(lambda: defaultdict(float))
    edge_agg_lat = defaultdict(lambda: defaultdict(float))
    agg_core_bw  = defaultdict(float)
    agg_core_lat = defaultdict(float)

    for lid, (src, dst) in topo_obj.links.items():
        bw  = topo_obj.link_bandwidths.get(lid, 0.0)
        lat = topo_obj.link_latencies.get(lid, 0.0)
        # edge → agg  (forward only; the reverse edge is skipped)
        if src in edge_set and dst in agg_set:
            edge_agg_bw[src][dst]  += bw
            edge_agg_lat[src][dst]  = lat
        # agg → core  (forward only)
        if src in agg_set and dst in core_set:
            agg_core_bw[src]  += bw
            agg_core_lat[src]  = lat

    # ── Group agg switches by which edge switches they connect to (= pod) ──
    agg_to_edges = {}   # agg_name → frozenset(edge_names)
    for agg in agg_set:
        agg_to_edges[agg] = frozenset(edge_agg_bw.keys() &
                                       {e for e in edge_agg_bw if agg in edge_agg_bw[e]})
    # invert: frozenset(edges) → [agg, ...]
    pod_map = defaultdict(list)
    for agg, edges_fs in agg_to_edges.items():
        pod_map[edges_fs].append(agg)

    pods = sorted(pod_map.values(), key=lambda ag: _get_num(ag[0]))

    # ── Build approximated link dict ──
    new_links = {}
    new_bw    = {}
    new_lat   = {}
    ctr = 1

    # Copy all links that don't touch agg or core nodes
    for lid, (src, dst) in topo_obj.links.items():
        if src in agg_set or src in core_set or dst in agg_set or dst in core_set:
            continue
        new_links[ctr] = (src, dst)
        new_bw[ctr]    = topo_obj.link_bandwidths.get(lid, 0.0)
        new_lat[ctr]   = topo_obj.link_latencies.get(lid, 0.0)
        ctr += 1

    # Add merged upper fabric
    upper_switches  = []
    tor_to_group    = {}
    group_to_core   = {}
    core_node = 'r0'
    upper_switches.append(core_node)

    for pod_idx, agg_list in enumerate(pods):
        merged_agg = f'g{pod_idx}'
        upper_switches.append(merged_agg)
        group_to_core[merged_agg] = core_node

        # Determine which edge switches belong to this pod
        pod_edges = sorted(pod_map[agg_to_edges[agg_list[0]]], key=_get_num)
        pod_edge_set = agg_to_edges[agg_list[0]]

        # edge ↔ merged_agg
        for edge_sw in sorted(pod_edge_set, key=_get_num):
            total_bw = sum(edge_agg_bw[edge_sw].get(a, 0.0) for a in agg_list)
            rep_lat  = next((edge_agg_lat[edge_sw][a]
                             for a in agg_list if a in edge_agg_lat[edge_sw]), 0.0)
            ctr = _add_bidir(new_links, new_bw, new_lat, ctr,
                             edge_sw, merged_agg, total_bw, rep_lat)
            tor_to_group[edge_sw] = merged_agg

        # merged_agg ↔ r0
        total_core_bw = sum(agg_core_bw.get(a, 0.0) for a in agg_list)
        rep_core_lat  = next((agg_core_lat[a] for a in agg_list
                              if a in agg_core_lat), 0.0)
        ctr = _add_bidir(new_links, new_bw, new_lat, ctr,
                         merged_agg, core_node, total_core_bw, rep_core_lat)

    metadata = {
        'tor_switches'  : frozenset(edge_set),
        'upper_switches': upper_switches,
        'tor_to_group'  : tor_to_group,
        'group_to_core' : group_to_core,
    }
    return new_links, new_bw, new_lat, metadata


# ---------------------------------------------------------------------------
# Dragonfly approximation
# ---------------------------------------------------------------------------

def approximate_dragonfly(topo_obj):
    """
    Approximate a Dragonfly topology.

    Original structure
    ------------------
    G groups × A switches/group (all switches ARE ToRs).
    - Intra-group links : full or partial mesh within each group
    - Inter-group links : h links per switch to other groups

    Approximation
    -------------
    Kept:
      - All NPU/NVSwitch/NIC nodes and their host links
      - All ToR switches (t*)
    Replaced:
      - Intra-group mesh    →  one 'g{i}' group aggregator per group
          BW(t_k → g_i)  = sum of all intra-group BW on switch t_k
      - All inter-group     →  one 'r0' global root
          BW(g_i → r0)   = sum of all inter-group BW across all switches in group i

    Communication paths
    -------------------
    Same-group  : NPU → ToR → g_i → ToR → NPU          (3 switch hops)
    Cross-group : NPU → ToR → g_src → r0 → g_dst → ToR → NPU  (5 switch hops)

    Args
    ----
    topo_obj : CustomizedDragonfly instance after DesignFullTopology() + LinksToG2ConfFile()

    Returns
    -------
    links (dict), link_bandwidths (dict), link_latencies (dict),
    metadata (dict):
        metadata['tor_switches']   – frozenset of all ToR switch names
        metadata['upper_switches'] – list of new upper-fabric switch names
        metadata['tor_to_group']   – dict  tor_name → group_switch_name
        metadata['group_to_core']  – dict  group_switch_name → 'r0'
    """
    A = topo_obj.num_switches       # switches per group
    G = topo_obj.num_groups         # number of groups
    N = topo_obj.total_num_switches # = G * A
    tor_names = {f't{i}' for i in range(1, N + 1)}

    # ── Classify switch-switch links as intra- or inter-group ──
    intra_bw  = defaultdict(float)   # tor_name → sum intra-group BW
    intra_lat = {}
    inter_bw  = defaultdict(float)   # tor_name → sum inter-group BW
    inter_lat = {}

    interpod_set = {(min(a, b), max(a, b)) for a, b in topo_obj.interpod_links}

    for lid, (src, dst) in topo_obj.links.items():
        if src not in tor_names or dst not in tor_names:
            continue
        bw  = topo_obj.link_bandwidths.get(lid, 0.0)
        lat = topo_obj.link_latencies.get(lid, 0.0)
        si  = _get_num(src) - 1   # 0-based
        di  = _get_num(dst) - 1
        pair = (min(si, di), max(si, di))
        if pair in interpod_set:
            inter_bw[src]  += bw
            inter_lat[src]  = lat
        else:
            intra_bw[src]  += bw
            intra_lat[src]  = lat

    # ── Build approximated links ──
    new_links = {}
    new_bw    = {}
    new_lat   = {}
    ctr = 1

    # Copy all non-switch-switch links
    for lid, (src, dst) in topo_obj.links.items():
        if src in tor_names and dst in tor_names:
            continue
        new_links[ctr] = (src, dst)
        new_bw[ctr]    = topo_obj.link_bandwidths.get(lid, 0.0)
        new_lat[ctr]   = topo_obj.link_latencies.get(lid, 0.0)
        ctr += 1

    # Add merged upper fabric
    core_node    = 'r0'
    upper_switches = [core_node]
    tor_to_group   = {}
    group_to_core  = {}

    for g in range(G):
        group_agg = f'g{g}'
        upper_switches.append(group_agg)
        group_to_core[group_agg] = core_node

        g_inter_bw  = 0.0
        g_inter_lat = 0.0

        for a in range(A):
            tor = f't{g * A + a + 1}'   # 1-based
            tor_to_group[tor] = group_agg

            bw_i  = intra_bw.get(tor, 0.0)
            lat_i = intra_lat.get(tor, 0.0)
            if bw_i > 0:
                ctr = _add_bidir(new_links, new_bw, new_lat, ctr,
                                 tor, group_agg, bw_i, lat_i)

            g_inter_bw  += inter_bw.get(tor, 0.0)
            g_inter_lat  = inter_lat.get(tor, g_inter_lat)

        if g_inter_bw > 0:
            ctr = _add_bidir(new_links, new_bw, new_lat, ctr,
                             group_agg, core_node, g_inter_bw, g_inter_lat)

    metadata = {
        'tor_switches'  : frozenset(tor_names),
        'upper_switches': upper_switches,
        'tor_to_group'  : tor_to_group,
        'group_to_core' : group_to_core,
    }
    return new_links, new_bw, new_lat, metadata


# ---------------------------------------------------------------------------
# Jellyfish approximation
# ---------------------------------------------------------------------------

def approximate_jellyfish(topo_obj):
    """
    Approximate a Jellyfish topology.

    Original structure
    ------------------
    num_switches random-graph switches (ALL are ToRs).
    Each switch has `degree` switch-to-switch links and
    `num_hosts_per_switch` host links.

    Approximation
    -------------
    Kept:
      - All NPU/NVSwitch/NIC nodes and their host links
      - All ToR switches (t*)
    Replaced:
      - Entire random switch mesh  →  one 'r0' global root
          BW(t_i → r0) = sum of BW of all original switch-switch links on t_i
                       = (avg links on t_i) × bw_switch_switch

    Communication path
    ------------------
    Any NPU pair : NPU_src → ToR_src → r0 → ToR_dst → NPU_dst  (2 switch hops)

    Note: because Jellyfish has no natural grouping, a single global root is
    the minimal tree that preserves total cross-switch bandwidth per node.

    Args
    ----
    topo_obj : Jellyfish instance after DesignFullTopology() + LinksToG2ConfFile()

    Returns
    -------
    links (dict), link_bandwidths (dict), link_latencies (dict),
    metadata (dict):
        metadata['tor_switches']   – frozenset of all ToR switch names
        metadata['upper_switches'] – ['r0']
        metadata['tor_to_group']   – dict  tor_name → 'r0'
        metadata['group_to_core']  – {}  (only 1 level above ToR)
    """
    tor_names = {f't{i}' for i in range(1, topo_obj.num_switches + 1)}

    # Sum switch-switch BW per ToR switch
    sw_bw_total = defaultdict(float)
    sw_lat_rep  = {}

    for lid, (src, dst) in topo_obj.links.items():
        if src in tor_names and dst in tor_names:
            bw  = topo_obj.link_bandwidths.get(lid, 0.0)
            lat = topo_obj.link_latencies.get(lid, 0.0)
            sw_bw_total[src] += bw
            sw_lat_rep[src]   = lat

    # ── Build approximated links ──
    new_links = {}
    new_bw    = {}
    new_lat   = {}
    ctr = 1

    # Copy all non-switch-switch links
    for lid, (src, dst) in topo_obj.links.items():
        if src in tor_names and dst in tor_names:
            continue
        new_links[ctr] = (src, dst)
        new_bw[ctr]    = topo_obj.link_bandwidths.get(lid, 0.0)
        new_lat[ctr]   = topo_obj.link_latencies.get(lid, 0.0)
        ctr += 1

    # Add star links to global root
    core_node    = 'r0'
    tor_to_group = {}

    for tor in sorted(tor_names, key=_get_num):
        bw  = sw_bw_total.get(tor, 0.0)
        lat = sw_lat_rep.get(tor, 0.0)
        if bw > 0:
            ctr = _add_bidir(new_links, new_bw, new_lat, ctr,
                             tor, core_node, bw, lat)
        tor_to_group[tor] = core_node

    metadata = {
        'tor_switches'  : frozenset(tor_names),
        'upper_switches': [core_node],
        'tor_to_group'  : tor_to_group,
        'group_to_core' : {},
    }
    return new_links, new_bw, new_lat, metadata


# ---------------------------------------------------------------------------
# Shared path generator (works for all 3 approximated topologies)
# ---------------------------------------------------------------------------

def generate_approx_paths(topo_obj, new_links):
    """
    Generate host-to-host paths for an approximated topology.

    Because the upper fabric is now a tree, there is exactly ONE path between
    any pair of ToR switches. This function:
      1. Builds a DiGraph from new_links
      2. Computes ALL-PAIRS shortest paths on the switch subgraph
         (fast — only O(num_switches²) pairs, small tree)
      3. Expands switch-level paths to full NPU-level paths using the same
         intra-node path logic as the original topology

    Yields
    ------
    (h_src, {h_dst: [[path]]}) for each source NPU
    """
    # Build graph
    G = nx.DiGraph()
    for src, dst in new_links.values():
        G.add_edge(src, dst)

    # All nodes that are NPUs
    total_npus = int(topo_obj.total_num_hosts)

    # Pre-compute switch-level all-pairs paths (graph is tiny)
    switch_paths = dict(nx.all_pairs_shortest_path(G))

    intra_topology = topo_obj.intra_node_topology
    npus_per_node  = topo_obj.npus_per_node

    for i in range(total_npus):
        h_src = f'h{i + 1}'
        if h_src not in topo_obj.npu_to_node:
            continue
        src_node = topo_obj.npu_to_node[h_src]
        src_tor  = topo_obj.node_to_switch.get(src_node)
        src_nvs  = topo_obj.npu_to_intra_switches.get(h_src, [])

        dests = {}
        for j in range(total_npus):
            if i == j:
                continue
            h_dst = f'h{j + 1}'
            if h_dst not in topo_obj.npu_to_node:
                continue
            dst_node = topo_obj.npu_to_node[h_dst]
            dst_tor  = topo_obj.node_to_switch.get(dst_node)
            dst_nvs  = topo_obj.npu_to_intra_switches.get(h_dst, [])

            if src_node == dst_node and intra_topology and npus_per_node > 1:
                # Same node: use intra-node path unchanged
                dests[h_dst] = topo_obj.generate_intra_node_paths(h_src, h_dst)

            elif src_tor == dst_tor:
                # Same ToR, different nodes — go through ToR (no upper fabric)
                snv = src_nvs[0] if src_nvs else None
                dnv = dst_nvs[0] if dst_nvs else None
                if snv and dnv:
                    dests[h_dst] = [[h_src, snv, src_tor, dnv, h_dst]]
                elif snv:
                    dests[h_dst] = [[h_src, snv, src_tor, h_dst]]
                elif dnv:
                    dests[h_dst] = [[h_src, src_tor, dnv, h_dst]]
                else:
                    dests[h_dst] = [[h_src, src_tor, h_dst]]

            else:
                # Different ToRs — look up switch-level path
                try:
                    sw_path = switch_paths[src_tor][dst_tor]
                except KeyError:
                    dests[h_dst] = [[h_src, h_dst]]
                    continue

                snv = src_nvs[0] if src_nvs else None
                dnv = dst_nvs[0] if dst_nvs else None
                if snv and dnv:
                    full = [h_src, snv] + sw_path + [dnv, h_dst]
                elif snv:
                    full = [h_src, snv] + sw_path + [h_dst]
                elif dnv:
                    full = [h_src] + sw_path + [dnv, h_dst]
                else:
                    full = [h_src] + sw_path + [h_dst]
                dests[h_dst] = [full]

        yield h_src, dests
