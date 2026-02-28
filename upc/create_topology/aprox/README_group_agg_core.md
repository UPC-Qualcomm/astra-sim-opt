# Approximated Topology Generator

This folder provides bandwidth-preserving approximations of FoldedClos,
Dragonfly, and Jellyfish topologies.  The output files are in the same
formats as the original `create_topology_main.py` (NS3 file, `.json`, `.txt`,
`_nodemap.json`) and can be used as drop-in replacements.

---

## Core Idea

The key insight is that the **upper fabric** of any datacenter network
(aggregation / core layers for FoldedClos; inter-group fabric for Dragonfly;
the random mesh for Jellyfish) can be **collapsed into a compact tree** without
changing what matters most for power/performance modelling:

- Total available bandwidth between any two ToR switches is preserved
  (link BW on each new link = **sum** of all original link BWs it replaces).
- The NPU count, NVSwitch count, and ToR switch count are **unchanged**.
- Because the approximated upper fabric is a tree, routing is **unique and
  deterministic** — no ECMP choices are needed.
- The number of switches drops dramatically, making topology files smaller
  and simulation / congestion models faster.

---

## FoldedClos Approximation

### Original structure (K-port fat-tree)

```
Core switches  (K/2)²              ← level 3  (K²/4 switches)
Agg  switches  K/2 per pod × K pods ← level 2  (K²/2 switches)
Edge / ToR     K/2 per pod × K pods ← level 1  (K²/2 switches)
NPUs                                ← level 0
```

For K=4: 4 core, 8 agg, 8 edge, 16 servers × N NPUs each.

### Approximated structure

```
r0   (1 merged core)              ← level 3  (1 switch)
g0..g{K-1}  (1 merged agg/pod)   ← level 2  (K switches)
Edge / ToR  (unchanged)           ← level 1  (K²/2 switches)
NPUs / NVSwitches (unchanged)     ← level 0
```

For K=4: **1 core + 4 agg + 8 edge** instead of 4+8+8.

### Bandwidth aggregation

| Link | Merged BW formula |
|------|-------------------|
| `edge_sw → g{pod}` | `bw_edge_agg × (number of agg switches in pod)` |
| `g{pod} → r0` | `bw_agg_core × (number of core links into that pod)` |

For K=4 with `bw_edge_agg = 100 GB/s`:
- Each pod has 2 agg switches; each edge switch has 1 link to each agg switch.
- `BW(edge → merged_agg)` = 100 × 2 = **200 GB/s** per edge switch.
- Each pod's 2 agg switches each connect to 2 core switches at 100 GB/s.
- `BW(merged_agg → r0)` = 100 × 2 × 2 = **400 GB/s**.

> Only the forward direction (edge→agg, agg→core) is counted so each
> physical link contributes its BW exactly once.

### Routing paths (approximated)

| Source / Dest | Path |
|---------------|------|
| Same node | intra-node path (NVSwitch, unchanged) |
| Same edge switch | `h → [NVSwitch] → ToR → [NVSwitch] → h` |
| Same pod, different edge | `h → [NVSwitch] → ToR_src → g{pod} → ToR_dst → [NVSwitch] → h` |
| Different pod | `h → [NVSwitch] → ToR_src → g{src_pod} → r0 → g{dst_pod} → ToR_dst → [NVSwitch] → h` |

---

## Dragonfly Approximation

### Original structure

```
G groups × A switches/group   ← ALL switches are ToRs
  - Intra-group: full/partial mesh within each group
  - Inter-group: h links per switch to switches in other groups
```

For G=4, A=4, h=2: 16 switches (all ToRs), ≈24 intra-group + 32 inter-group links.

### Approximated structure

```
r0     (1 global root)            ← level 2  (1 switch)
g0..g{G-1}  (1 group aggregator per group)  ← level 1  (G switches)
ToR switches (unchanged, t*)      ← level 0  (G×A switches)
NPUs / NVSwitches (unchanged)
```

### Bandwidth aggregation

| Link | Merged BW formula |
|------|-------------------|
| `t_k → g{group}` | Sum of all **intra-group** link BWs on switch `t_k` |
| `g{group} → r0` | Sum of all **inter-group** link BWs across all A switches in the group |

Example: G=4, A=4, `bw_intra_group=200`, `bw_inter_group=100`, h=2.
- Each ToR has 3 intra-group neighbours at 200 GB/s → `BW(t_k → g)` = **600 GB/s**.
- Each group has 4 switches × h=2 inter-group links × 100 GB/s → `BW(g → r0)` = **800 GB/s**.

### Routing paths (approximated)

| Source / Dest | Path |
|---------------|------|
| Same node | intra-node path (unchanged) |
| Same group | `h → [NVSwitch] → ToR_src → g{group} → ToR_dst → [NVSwitch] → h` |
| Different group | `h → [NVSwitch] → ToR_src → g{src} → r0 → g{dst} → ToR_dst → [NVSwitch] → h` |

---

## Jellyfish Approximation

### Original structure

```
num_switches random-graph switches  ← ALL are ToRs
  - Each switch: `degree` switch-switch links + num_hosts_per_switch host links
```

### Approximated structure

```
r0   (1 global root — star centre)  ← level 1  (1 switch)
ToR switches (unchanged, t*)        ← level 0  (num_switches switches)
NPUs / NVSwitches (unchanged)
```

Because Jellyfish has no natural group structure, the minimal
bandwidth-preserving tree above the ToRs is a simple **star**.

### Bandwidth aggregation

| Link | Merged BW formula |
|------|-------------------|
| `t_i → r0` | Sum of BW on all switch-switch links of `t_i` = `actual_links_on_t_i × bw_switch_switch` |

Example: degree=3, `bw_switch_switch=200` → `BW(t_i → r0)` = **600 GB/s**.

### Routing paths (approximated)

| Source / Dest | Path |
|---------------|------|
| Same node | intra-node path (unchanged) |
| Same ToR | `h → [NVSwitch] → ToR → [NVSwitch] → h` |
| Different ToR | `h → [NVSwitch] → ToR_src → r0 → ToR_dst → [NVSwitch] → h` |

---

## Switch count comparison

| Topology | Original switches | Approximated switches |
|----------|:-----------------:|:---------------------:|
| FoldedClos K=4 | 4+8+8 = 20 | 1+4+8 = **13** |
| FoldedClos K=16 | 256+512+512 = 1280 | 1+16+512 = **529** |
| Dragonfly G=4, A=4 | 16 | 1+4+16 = **21** |
| Jellyfish 8 switches | 8 | 1+8 = **9** |

> Note: Dragonfly appears to *increase* switch count because the original 16
> switches are all ToRs and 5 new upper-fabric nodes are added.  The gain is
> in removing all O(G²·A²/2) switch-switch links and replacing them with
> O(G·A + G) links, which dramatically reduces topology file size and
> routing complexity for large G/A values.

---

## Usage

```python
from generate_approx import generate_approx_topology_files

# FoldedClos K=16, 8 NPUs/node, 4 NVSwitches
fc_bw = {'host_edge': 200, 'edge_agg': 200, 'agg_core': 200, 'intra_node': 900}
generate_approx_topology_files('FoldedClos', {
    'K': 16,
    'bandwidth_config': fc_bw,
    'bw_unit': 'GB/s',
    'npus_per_node': 8,
    'intra_node_topology': 'switch',
    'num_nvswitches': 4,
}, output_dir='./output', base_filename='FoldedClos_K16_approx')

# Dragonfly G=8, A=8, h=4, 8 NPUs/node
df_bw = {'host_switch': 200, 'intra_group': 400, 'inter_group': 200, 'intra_node': 900}
generate_approx_topology_files('Dragonfly', {
    'G': 8, 'A': 8, 'h': 4, 'concentration': 1,
    'bandwidth_config': df_bw, 'bw_unit': 'GB/s',
    'npus_per_node': 8, 'intra_node_topology': 'switch', 'num_nvswitches': 4,
}, output_dir='./output', base_filename='Dragonfly_G8A8_approx')

# Jellyfish 64 switches, degree 6, 8 NPUs/node
jf_bw = {'host_switch': 200, 'switch_switch': 400, 'intra_node': 900}
generate_approx_topology_files('Jellyfish', {
    'num_switches': 64, 'degree': 6, 'num_hosts_per_switch': 1,
    'bandwidth_config': jf_bw, 'bw_unit': 'GB/s',
    'npus_per_node': 8, 'intra_node_topology': 'switch', 'num_nvswitches': 4,
}, output_dir='./output', base_filename='Jellyfish_64sw_approx')
```

Run the built-in self-test:
```bash
cd aprox/
python generate_approx.py
```

---

## File structure

```
aprox/
├── aprox_topology.py     # Approximation logic for all 3 topologies + path generator
├── generate_approx.py    # Main entry point (equivalent to create_topology_main.py)
└── README.md             # This file
```

Output files follow the same naming and format as the original generator:
```
{base_filename}               ← NS3 format (plain integer node IDs)
{base_filename}.json          ← G2 format (prefixed node names, full edge list)
{base_filename}.txt           ← G2 text format
{base_filename}_nodemap.json  ← integer ID → prefixed name mapping
```
