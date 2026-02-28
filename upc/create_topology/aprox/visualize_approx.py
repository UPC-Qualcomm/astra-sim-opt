"""
visualize_approx.py
===================
Visualization helpers for approximated topologies produced by
approximate_foldedclos / approximate_dragonfly / approximate_jellyfish.

Public API
----------
  visualize_approx_topology(topo_obj, new_links, new_bw, metadata, ...)
      Full-topology view of the approximated network.

  visualize_approx_single_server(topo_obj, new_links, new_bw, metadata,
                                  server_idx, ...)
      Stack view for one server: NPUs → NVSwitches → ToR → upper fabric.

Node type ↔ colour mapping
--------------------------
  h*          NPU              green    #4CAF50
  v*          NVSwitch         orange   #FF9800
  n* / p*     NIC / NPU-sw     pink     #E91E63
  t*          ToR (unchanged)  blue     #2196F3
  g*          Merged agg       purple   #9C27B0
  c*          Core (unchanged) red      #F44336
  r0          Root switch      dark-red #B71C1C
"""

import re
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_COLORS = {
    'npu':        '#4CAF50',   # green
    'nvswitch':   '#FF9800',   # orange
    'npu_switch': '#FF9800',   # orange  (p* internal per-NPU switches)
    'nic':        '#E91E63',   # pink    (n* NIC switches for ring/fc)
    'tor_switch': '#2196F3',   # blue
    'merged_agg': '#9C27B0',   # purple
    'core_switch':'#F44336',   # red
    'root':       '#B71C1C',   # dark red
}

_LEGEND_LABELS = {
    'npu':        'NPU (h*)',
    'nvswitch':   'NVSwitch (v*)',
    'npu_switch': 'NPU-Switch (p*)',
    'nic':        'NIC Switch (n*)',
    'tor_switch': 'ToR Switch (t*)  — unchanged',
    'merged_agg': 'Merged Aggregator (g*)  — approximated',
    'core_switch':'Core Switch (c*)  — unchanged',
    'root':       'Root Switch (r0)  — approximated',
}


# ---------------------------------------------------------------------------
# Node classification
# ---------------------------------------------------------------------------

def _classify(name):
    """
    Return the node type string for *name* based purely on its prefix.
    Works for both original and approximated nodes.
    """
    if name.startswith('h'):
        return 'npu'
    if name.startswith('v'):
        return 'nvswitch'
    if name.startswith('p'):
        return 'npu_switch'
    if name.startswith('n'):
        return 'nic'
    if name == 'r0':
        return 'root'
    if name.startswith('g'):
        return 'merged_agg'
    if name.startswith('c'):
        return 'core_switch'
    if name.startswith('t'):
        return 'tor_switch'
    return 'tor_switch'   # fallback


def _level(name, has_merged_agg, has_upper):
    """
    Return y-level (0 = bottom) for layout.

    Levels:
      0  NPU
      1  NVSwitch / NPU-switch
      2  NIC
      3  ToR
      4  Merged agg (g*)   — or root if no g*
      5  Core / Root        — only when both g* and (core or r0) exist
    """
    cls = _classify(name)
    base = {'npu': 0, 'nvswitch': 1, 'npu_switch': 1, 'nic': 2, 'tor_switch': 3}
    if cls in base:
        return base[cls]
    if cls == 'merged_agg':
        return 4
    # core_switch or root
    if has_merged_agg:
        return 5
    return 4   # no g* layer → root/core sits right above ToR


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def _make_layout(G, width=12, height=10):
    """
    Build a hierarchical x/y position dict for all nodes in *G*.
    Nodes at the same level are spaced evenly across *width*.
    """
    has_merged = any(_classify(n) == 'merged_agg' for n in G.nodes())
    has_upper  = any(_classify(n) in ('core_switch', 'root') for n in G.nodes())

    # group by level
    by_level = {}
    for n in G.nodes():
        lvl = _level(n, has_merged, has_upper)
        by_level.setdefault(lvl, []).append(n)

    num_levels = max(by_level.keys()) + 1 if by_level else 1

    def _sort_key(name):
        m = re.search(r'\d+', name)
        return (name[0], int(m.group()) if m else 0)

    pos = {}
    for lvl, nodes in by_level.items():
        nodes_s = sorted(nodes, key=_sort_key)
        n = len(nodes_s)
        for i, node in enumerate(nodes_s):
            x = (i + 0.5) * width / n - width / 2
            y = lvl * height / max(num_levels - 1, 1)
            pos[node] = (x, y)

    return pos


# ---------------------------------------------------------------------------
# Full-topology visualisation
# ---------------------------------------------------------------------------

def visualize_approx_topology(topo_obj, new_links, new_bw, metadata,
                               title="Approximated Topology",
                               figsize=(16, 12), show_bw=True):
    """
    Draw the complete approximated topology.

    Parameters
    ----------
    topo_obj   : original topology object (only used for its name/type info)
    new_links  : dict {id: (src, dst)}   returned by approximate_*()
    new_bw     : dict {id: bw}           returned by approximate_*()
    metadata   : dict                    returned by approximate_*()
    title      : plot title
    figsize    : (width, height) in inches
    show_bw    : label every edge with its bandwidth value

    Returns
    -------
    fig, ax, G  (matplotlib Figure, Axes, DiGraph)
    """
    G = nx.DiGraph()
    for lid, (src, dst) in new_links.items():
        bw = new_bw.get(lid, 1.0)
        if G.has_edge(src, dst):
            G[src][dst]['bandwidth'] = max(G[src][dst]['bandwidth'], bw)
        else:
            G.add_edge(src, dst, bandwidth=bw)

    pos = _make_layout(G)

    node_colors = [_COLORS.get(_classify(n), '#757575') for n in G.nodes()]

    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=600, ax=ax)

    edges = list(G.edges())
    if edges:
        bws = [G[u][v]['bandwidth'] for u, v in edges]
        max_bw = max(bws) if bws else 1
        widths = [1 + 3 * b / max_bw for b in bws]
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
                               width=widths, alpha=0.7, ax=ax,
                               connectionstyle='arc3,rad=0.1')

    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)

    if show_bw:
        elabels = {(u, v): f"{G[u][v]['bandwidth']:.0f}"
                   for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, elabels,
                                     font_size=6, ax=ax, label_pos=0.3)

    # Legend — only for node types actually present
    present = {_classify(n) for n in G.nodes()}
    legend_order = ['npu', 'nvswitch', 'npu_switch', 'nic',
                    'tor_switch', 'merged_agg', 'core_switch', 'root']
    handles = [mpatches.Patch(color=_COLORS[t], label=_LEGEND_LABELS[t])
               for t in legend_order if t in present]
    ax.legend(handles=handles, loc='upper right', fontsize=9)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    return fig, ax, G


# ---------------------------------------------------------------------------
# Single-server visualisation
# ---------------------------------------------------------------------------

def visualize_approx_single_server(topo_obj, new_links, new_bw, metadata,
                                    server_idx=1, figsize=(14, 10)):
    """
    Show the full vertical stack for one server in the approximated topology.

    The view spans from NPUs at the bottom up through NVSwitches → ToR →
    merged agg (g*) → core / root.  Approximated nodes are highlighted with
    a dashed border around the upper-fabric section.

    Parameters
    ----------
    topo_obj   : original topology object
    new_links  : dict {id: (src, dst)} from approximate_*()
    new_bw     : dict {id: bw}         from approximate_*()
    metadata   : dict                  from approximate_*()
    server_idx : 1-based server index
    figsize    : (width, height) in inches

    Returns
    -------
    fig, ax
    """
    if not hasattr(topo_obj, 'server_nodes') or \
            server_idx not in topo_obj.server_nodes:
        avail = list(getattr(topo_obj, 'server_nodes', {}).keys())[:5]
        print(f"Server {server_idx} not found. Available: {avail}...")
        return None, None

    # ── Nodes belonging to this server ────────────────────────────────────
    node_ids = topo_obj.server_nodes[server_idx]
    main_sw  = (topo_obj.server_to_switch.get(server_idx)
                if hasattr(topo_obj, 'server_to_switch') else None)

    all_npus      = set()
    all_nvswitches = set()
    for nid in node_ids:
        npus = topo_obj.node_npus.get(nid, [])
        all_npus.update(npus)
        for npu in npus:
            all_nvswitches.update(topo_obj.npu_to_intra_switches.get(npu, []))

    # ── Upper-fabric nodes reachable from this ToR (2 hops up) ───────────
    upper = set()
    if main_sw:
        upper.add(main_sw)
        for lid, (src, dst) in new_links.items():
            if src == main_sw and not dst.startswith('h'):
                upper.add(dst)
            elif dst == main_sw and not src.startswith('h'):
                upper.add(src)
        # one more hop (g* → core / root)
        frontier = set(upper)
        for n in frontier:
            for lid, (src, dst) in new_links.items():
                if src == n and _classify(dst) in ('core_switch', 'root'):
                    upper.add(dst)
                elif dst == n and _classify(src) in ('core_switch', 'root'):
                    upper.add(src)

    relevant = all_npus | all_nvswitches | upper

    # ── Build subgraph ─────────────────────────────────────────────────────
    G = nx.DiGraph()
    for lid, (src, dst) in new_links.items():
        if src in relevant and dst in relevant:
            bw = new_bw.get(lid, 1.0)
            if G.has_edge(src, dst):
                G[src][dst]['bandwidth'] = max(G[src][dst]['bandwidth'], bw)
            else:
                G.add_edge(src, dst, bandwidth=bw)

    pos = _make_layout(G, width=10, height=8)

    node_colors = [_COLORS.get(_classify(n), '#757575') for n in G.nodes()]

    fig, ax = plt.subplots(figsize=figsize)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                           node_size=1400, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

    edges = list(G.edges())
    if edges:
        bws = [G[u][v]['bandwidth'] for u, v in edges]
        max_bw = max(bws) if bws else 1
        widths = [1 + 4 * b / max_bw for b in bws]
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
                               width=widths, alpha=0.7, ax=ax,
                               connectionstyle='arc3,rad=0.05')

    elabels = {(u, v): f"{G[u][v]['bandwidth']:.0f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, elabels, font_size=9, ax=ax)

    # Dashed box around approximated upper-fabric nodes
    approx_nodes = {n for n in G.nodes()
                    if _classify(n) in ('merged_agg', 'root')}
    if approx_nodes and pos:
        xs = [pos[n][0] for n in approx_nodes]
        ys = [pos[n][1] for n in approx_nodes]
        pad = 0.6
        rect = plt.Rectangle(
            (min(xs) - pad, min(ys) - pad),
            (max(xs) - min(xs) + 2 * pad),
            (max(ys) - min(ys) + 2 * pad),
            fill=False, linestyle='--', edgecolor='#9C27B0', linewidth=2
        )
        ax.add_patch(rect)
        ax.text(min(xs) - pad + 0.1, max(ys) + pad - 0.1,
                'Approximated\nupper fabric',
                fontsize=9, color='#9C27B0', va='top')

    # Legend
    present = {_classify(n) for n in G.nodes()}
    legend_order = ['npu', 'nvswitch', 'npu_switch', 'nic',
                    'tor_switch', 'merged_agg', 'core_switch', 'root']
    handles = [mpatches.Patch(color=_COLORS[t], label=_LEGEND_LABELS[t])
               for t in legend_order if t in present]
    ax.legend(handles=handles, loc='upper right', fontsize=9)

    ax.set_title(
        f"Server {server_idx}  —  NPUs → NVSwitches → ToR → Approximated Upper Fabric",
        fontsize=13, fontweight='bold'
    )
    ax.axis('off')
    plt.tight_layout()
    return fig, ax
