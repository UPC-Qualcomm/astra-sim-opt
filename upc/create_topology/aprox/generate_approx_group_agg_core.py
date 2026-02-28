"""
generate_approx.py
==================
Main entry point for generating approximated topology files.

Usage
-----
    from generate_approx import generate_approx_topology_files

    # FoldedClos example
    fc_bw = {'host_edge': 200, 'edge_agg': 200, 'agg_core': 200, 'intra_node': 900}
    fc_config = {
        'K': 16,
        'bandwidth_config': fc_bw,
        'bw_unit': 'GB/s',
        'npus_per_node': 8,
        'intra_node_topology': 'switch',
        'num_nvswitches': 4,
    }
    generate_approx_topology_files('FoldedClos', fc_config,
                                   base_filename='FoldedClos_approx_8192npus')

The function produces the same output file formats as generate_topology_files
in create_topology_main.py (NS3 file, .json, .txt) but with the approximated
upper fabric.
"""

import sys
import os

# ── Locate the 'new' folder which holds create_topology.py and utils.py ──
_HERE    = os.path.dirname(os.path.abspath(__file__))
_NEW_DIR = os.path.join(_HERE, '..', 'new')
if _NEW_DIR not in sys.path:
    sys.path.insert(0, _NEW_DIR)

from create_topology import CustomizedDragonfly, Jellyfish, FoldedClos
from utils import write_ns3_topology_file, write_g2_topology_files
from aprox_topology import (
    approximate_foldedclos,
    approximate_dragonfly,
    approximate_jellyfish,
    generate_approx_paths,
)


def generate_approx_topology_files(topology, config,
                                   output_dir='./',
                                   base_filename=None):
    """
    Build the original topology, approximate its upper fabric, then write
    NS3 and G2 topology files.

    Parameters
    ----------
    topology : str
        'FoldedClos', 'Dragonfly', or 'Jellyfish'
    config : dict
        Same config dict accepted by generate_topology_files in
        create_topology_main.py.
    output_dir : str
        Directory where output files are written (default: current dir).
    base_filename : str or None
        Base name for output files.  Defaults to '{topology}_approx'.
    """
    print(f"--- Generating APPROXIMATED {topology} topology ---")

    if base_filename is None:
        base_filename = f'{topology}_approx'

    bw_config  = config.get('bandwidth_config', {})
    lat_config = config.get('latency_config', {})
    bw_unit    = config.get('bw_unit', 'GB/s')
    lat_unit   = config.get('lat_unit', 'ms')

    nodes_per_server   = config.get('nodes_per_server', 1)
    npus_per_node      = config.get('npus_per_node', 1)
    intra_node_topology = config.get('intra_node_topology', None)
    num_nvswitches     = config.get('num_nvswitches', 1)

    # ── 1. Build original topology object ──────────────────────────────────
    if topology == 'FoldedClos':
        K = config['K']
        total_servers = int(K**3 / 4)
        total_npus    = total_servers * nodes_per_server * npus_per_node
        print(f"  Original: K={K}, servers={total_servers}, NPUs={total_npus}")
        topo_obj = FoldedClos(K, 1, 1,
                              bandwidth_config=bw_config,
                              latency_config=lat_config,
                              nodes_per_server=nodes_per_server,
                              npus_per_node=npus_per_node,
                              intra_node_topology=intra_node_topology,
                              num_nvswitches=num_nvswitches)

    elif topology == 'Dragonfly':
        G  = config['G']
        A  = config['A']
        h  = config.get('h', 1)
        c  = config.get('concentration', 1)
        total_servers = G * A * c
        total_npus    = total_servers * nodes_per_server * npus_per_node
        print(f"  Original: G={G}, A={A}, h={h}, conc={c}, "
              f"servers={total_servers}, NPUs={total_npus}")
        topo_obj = CustomizedDragonfly(G, A, h, c,
                                       bandwidth_config=bw_config,
                                       latency_config=lat_config,
                                       nodes_per_server=nodes_per_server,
                                       npus_per_node=npus_per_node,
                                       intra_node_topology=intra_node_topology,
                                       num_nvswitches=num_nvswitches)

    elif topology == 'Jellyfish':
        num_sw    = config['num_switches']
        degree    = config['degree']
        hps       = config.get('num_hosts_per_switch', 1)
        total_servers = num_sw * hps
        total_npus    = total_servers * nodes_per_server * npus_per_node
        print(f"  Original: switches={num_sw}, degree={degree}, "
              f"hosts/switch={hps}, NPUs={total_npus}")
        topo_obj = Jellyfish(num_sw, degree,
                             num_hosts_per_switch=hps,
                             bandwidth_config=bw_config,
                             latency_config=lat_config,
                             nodes_per_server=nodes_per_server,
                             npus_per_node=npus_per_node,
                             intra_node_topology=intra_node_topology,
                             num_nvswitches=num_nvswitches)
    else:
        raise ValueError(f"Unknown topology: {topology!r}")

    # ── 2. Design topology (builds adjacency matrix + switch-switch links) ─
    topo_obj.DesignFullTopology()

    # ── 3. Add host-facing links (NPUs, NVSwitches, NICs → ToR) ───────────
    links_out = topo_obj.LinksToG2ConfFile()
    if isinstance(links_out, tuple):
        links_out = links_out[0]  # FoldedClos returns (links, str)

    # ── 4. Apply approximation ─────────────────────────────────────────────
    if topology == 'FoldedClos':
        approx_fn = approximate_foldedclos
    elif topology == 'Dragonfly':
        approx_fn = approximate_dragonfly
    else:
        approx_fn = approximate_jellyfish

    new_links, new_bw, new_lat, metadata = approx_fn(topo_obj)

    n_orig_sw  = _count_switches(topo_obj)
    n_approx_sw = len(metadata['upper_switches']) + len(metadata['tor_switches'])
    print(f"  Switches: {n_orig_sw} original  →  {n_approx_sw} approximated")
    print(f"    Upper-fabric nodes: {metadata['upper_switches']}")

    # ── 5. Generate paths (streaming generator — no O(N²) memory) ─────────
    path_gen = lambda: generate_approx_paths(topo_obj, new_links)

    # ── 6. Write files ─────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Writing files to: {output_dir}")

    write_g2_topology_files(new_links, None,
                            base_filename=base_filename,
                            link_bandwidths=new_bw,
                            link_latencies=new_lat,
                            output_dir=output_dir,
                            path_gen=path_gen)

    write_ns3_topology_file(new_links, None,
                            filename=base_filename,
                            link_bandwidths=new_bw,
                            link_latencies=new_lat,
                            bw_unit=bw_unit,
                            lat_unit=lat_unit,
                            output_dir=output_dir,
                            path_gen=path_gen)

    print(f"--- Done: {base_filename} ---\n")


def _count_switches(topo_obj):
    """Return total number of switch nodes in the original topology."""
    all_nodes = set()
    for src, dst in topo_obj.links.values():
        all_nodes.add(src)
        all_nodes.add(dst)
    return sum(1 for n in all_nodes if not n.startswith('h'))


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    out = './approx_test_output'

    print("=" * 60)
    print("TEST 1 — FoldedClos K=4, 2 NPUs/node, switch intra-node")
    print("=" * 60)
    fc_bw = {'host_edge': 100, 'edge_agg': 100, 'agg_core': 100, 'intra_node': 900}
    generate_approx_topology_files('FoldedClos', {
        'K': 14, 'bandwidth_config': fc_bw, 'bw_unit': 'GB/s',
        'npus_per_node': 8, 'intra_node_topology': 'switch', 'num_nvswitches': 1,
    }, output_dir=out, base_filename='FoldedClos_K4_approx')

    #print("=" * 60)
    #print("TEST 2 — Dragonfly G=4, A=4, h=2, 2 NPUs/node")
    #print("=" * 60)
    #df_bw = {'host_switch': 100, 'intra_group': 200, 'inter_group': 100, 'intra_node': 900}
    #generate_approx_topology_files('Dragonfly', {
    #    'G': 4, 'A': 4, 'h': 2, 'concentration': 1,
    #    'bandwidth_config': df_bw, 'bw_unit': 'GB/s',
    #    'npus_per_node': 2, 'intra_node_topology': 'switch', 'num_nvswitches': 1,
    #}, output_dir=out, base_filename='Dragonfly_G4A4_approx')
#
    #print("=" * 60)
    #print("TEST 3 — Jellyfish 8 switches, degree 3, 2 NPUs/node")
    #print("=" * 60)
    #jf_bw = {'host_switch': 100, 'switch_switch': 200, 'intra_node': 900}
    #generate_approx_topology_files('Jellyfish', {
    #    'num_switches': 8, 'degree': 3, 'num_hosts_per_switch': 1,
    #    'bandwidth_config': jf_bw, 'bw_unit': 'GB/s',
    #    'npus_per_node': 2, 'intra_node_topology': 'switch', 'num_nvswitches': 1,
    #}, output_dir=out, base_filename='Jellyfish_8sw_approx')
