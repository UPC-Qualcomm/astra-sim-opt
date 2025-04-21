#ifndef __MEMORY_HH__
#define __MEMORY_HH__

#include <algorithm>
#include <bits/stdc++.h>
#include <iostream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Common.hh"
#include "extern/graph_frontend/chakra/src/feeder/et_feeder.h"

namespace AstraSim {

class Memory {
  public:
    struct NodeInfo {
        uint64_t size;
        std::vector<uint64_t> child_node_ids;

        NodeInfo(uint64_t s = 0, const std::vector<uint64_t>& children = {})
            : size(s),
              child_node_ids(children) {}
    };

    struct NodeIndex {
        std::unordered_map<uint64_t, NodeInfo> nodes_;

        void addNode(uint64_t node_id,
                     uint64_t size,
                     const std::vector<uint64_t>& children) {
            nodes_[node_id] = NodeInfo(size, children);
        }

        uint64_t getSize(uint64_t node_id) const {
            auto it = nodes_.find(node_id);
            return (it != nodes_.end()) ? it->second.size : 0;
        }

        const std::vector<uint64_t>& getChildren(uint64_t node_id) const {
            static const std::vector<uint64_t> empty;
            auto it = nodes_.find(node_id);
            return (it != nodes_.end()) ? it->second.child_node_ids : empty;
        }

        bool contains(uint64_t node_id) const {
            return nodes_.find(node_id) != nodes_.end();
        }

        void clear() {
            nodes_.clear();
        }

        // Sum all node sizes
        uint64_t totalSize() const {
            uint64_t total = 0;
            for (const auto& [id, info] : nodes_) {
                if (info.child_node_ids.size() == 0) {
                    continue;
                }
                total += info.size;
            }
            return total;
        }

        // Replace a child ID with another one
        void replaceChildId(uint64_t old_id,
                            uint64_t new_id,
                            const std::vector<uint64_t>& candidate_node_ids) {

            for (auto node_id : candidate_node_ids) {
                auto it = nodes_.find(node_id);
                if (it != nodes_.end()) {
                    auto& children = it->second.child_node_ids;
                    std::replace(children.begin(), children.end(), old_id,
                                 new_id);
                }
            }
        }

        void replaceNodeWithChildrenInParents(
            uint64_t node_id,
            const std::vector<uint64_t>& new_children,
            const std::vector<uint64_t>& parent_ids) {
            for (auto parent_id : parent_ids) {
                auto it = nodes_.find(parent_id);
                if (it != nodes_.end()) {
                    auto& children = it->second.child_node_ids;

                    // Remove the node_id from the parent's children
                    children.erase(
                        std::remove(children.begin(), children.end(), node_id),
                        children.end());

                    // Add new children, avoiding duplicates
                    for (auto child : new_children) {
                        if (std::find(children.begin(), children.end(),
                                      child) == children.end()) {
                            children.push_back(child);
                        }
                    }
                }
            }
        }

        void replaceNodeWithChildrenEverywhere(
            uint64_t node_id, const std::vector<uint64_t>& new_children) {

            for (auto& [_, info] : nodes_) {
                auto& children = info.child_node_ids;

                // Check if node_id exists in this node's children
                auto it = std::find(children.begin(), children.end(), node_id);
                if (it != children.end()) {
                    // Remove the node_id
                    children.erase(
                        std::remove(children.begin(), children.end(), node_id),
                        children.end());

                    // Add new children, avoiding duplicates
                    for (auto child : new_children) {
                        if (std::find(children.begin(), children.end(),
                                      child) == children.end()) {
                            children.push_back(child);
                        }
                    }
                }
            }
        }

        // Remove a node entirely
        void removeNode(uint64_t node_id) {
            nodes_.erase(node_id);

            // Optionally, also remove from any parent node's children list
            removeChildIdFromAll(node_id);
        }

        void removeChildIdFromAll(uint64_t child_id) {
            for (auto& [_, info] : nodes_) {
                auto& children = info.child_node_ids;
                children.erase(
                    std::remove(children.begin(), children.end(), child_id),
                    children.end());
            }
        }

        void print(const std::string& title = "NodeIndex") const {
            std::cout << "====== " << title << " ======" << std::endl;
            for (const auto& [node_id, info] : nodes_) {
                std::cout << "Node ID: " << node_id << ", Size: " << info.size
                          << ", Children: [";
                for (size_t i = 0; i < info.child_node_ids.size(); ++i) {
                    std::cout << info.child_node_ids[i];
                    if (i != info.child_node_ids.size() - 1) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]" << std::endl;
            }
            std::cout << std::endl;
        }
    };
    enum LayerTypes : int { EMB, MHA, FFA };

    Memory();
    void set_memory_size(long long size);
    long long get_memory_size();
    long long get_consumed_memory();
    long long get_activation_memory();
    long long get_gradient_memory();
    long long get_parameter_memory();
    long long get_optimizer_memory();
    long long get_free_memory();

    int get_curr_stack(std::string node_name);
    std::string layer_type_to_str(LayerTypes type);
    Memory::LayerTypes str_to_layer_type(std::string node_name);
    bool check_free_memory(const std::shared_ptr<Chakra::ETFeederNode> node,
                           int sys_id);
    void update_consumed_memory(
        const std::shared_ptr<Chakra::ETFeederNode> node,
        int sys_id);

    ~Memory();

    long long memory_size;
    long long consumed_memory;
    NodeIndex activation_memory;
    long long parameter_memory;
    NodeIndex gradient_memory;
    long long optimizer_memory;
    long long tmp_activation_memory;
    long long tmp_gradient_memory;
    bool is_forward_pass;
    int layer_type_prev;
    int stop_recording_memory;
    int prev_stack_number;
    bool trace_mem;
    bool is_mixed_percision;
};

}  // namespace AstraSim

#endif /* __MEMORY_HH__ */
