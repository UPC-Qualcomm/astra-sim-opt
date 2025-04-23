#include "astra-sim/system/Memory.hh"

#include "astra-sim/system/LogGP.hh"
#include "astra-sim/system/Sys.hh"

using namespace std;
using namespace AstraSim;

typedef ChakraProtoMsg::NodeType ChakraNodeType;
typedef ChakraProtoMsg::CollectiveCommType ChakraCollectiveCommType;

Memory::Memory() {
    this->memory_size = 0;
    this->curr_consumed_mem = 0;
    this->max_consumed_memory = 0;
    this->max_parameter_memory = 0;
    this->max_optimizer_memory = 0;
    this->max_activation_memory = 0;
    this->max_gradient_memory = 0;
    this->acumilated_activation_memory = 0;
    this->acumilated_gradient_memory = 0;
    this->is_forward_pass = true;
    this->stop_recording_memory = false;
    this->is_mixed_percision = false;
    this->trace_mem = false;
    this->is_oom = false;
    this->layer_type_prev = Memory::LayerTypes::EMB;
    this->prev_stack_number = 0;
}

void Memory::set_memory_size(long long size) {
    this->memory_size = size;
}

long long Memory::get_memory_size() {
    return this->memory_size;
}

long long Memory::get_curr_consumed_memory() {
    return this->curr_consumed_mem;
}

long long Memory::get_max_consumed_memory() {
    this->max_consumed_memory = this->get_max_gradient_memory() +
                                this->get_max_parameter_memory() +
                                this->get_max_optimizer_memory();
    // this->max_consumed_memory =
    //     this->max_consumed_memory > mem ? this->max_consumed_memory : mem;

    return this->max_consumed_memory;
}

long long Memory::get_max_activation_memory() {
    long long act = this->get_curr_activation_memory();
    this->max_activation_memory =
        this->max_activation_memory > act ? this->max_activation_memory : act;
    return this->max_activation_memory;
}

long long Memory::get_max_gradient_memory() {
    long long act = this->get_curr_gradient_memory();
    this->max_gradient_memory =
        this->max_gradient_memory > act ? this->max_gradient_memory : act;
    return this->max_gradient_memory;
}

long long Memory::get_max_parameter_memory() {
    return this->max_parameter_memory;
}

long long Memory::get_max_optimizer_memory() {
    int opt_bytes = this->is_mixed_percision ? 2 * 3 : 2;
    return this->max_parameter_memory * opt_bytes;
}

long long Memory::get_curr_activation_memory() {
    return this->activation_memory.totalSize();
}

long long Memory::get_curr_gradient_memory() {
    return this->gradient_memory.totalSize();
}

long long Memory::get_acumilated_activation_memory() {
    return this->acumilated_activation_memory;
}

long long Memory::get_acumilated_gradient_memory() {
    return this->acumilated_gradient_memory;
}

long long Memory::get_free_memory() {
    long long free_mem = this->memory_size - this->get_curr_consumed_memory();
    return free_mem > 0 ? free_mem : 0;
}

int Memory::get_num_bytes() {
    return this->is_mixed_percision ? 2 : 4;
}

int Memory::get_is_oom() {
    return this->is_oom ? 1 : 0;
}

int Memory::get_curr_stack(std::string name) {
    std::regex pattern(R"(stack_(\d+)_)");
    std::smatch match;
    if (std::regex_search(name, match, pattern)) {
        return std::stoi(match[1]);
    }
    return -1;
}

std::string Memory::layer_type_to_str(Memory::LayerTypes type) {
    switch (type) {
    case Memory::LayerTypes::EMB:
        return "emb";
    case Memory::LayerTypes::MHA:
        return "mha";
    case Memory::LayerTypes::FFA:
        return "ffn";
    default:
        std::ostringstream oss;
        oss << "Error: Layer type is unknown: " << type;
        throw std::logic_error(oss.str());
    }
}

Memory::LayerTypes Memory::str_to_layer_type(std::string name) {
    if (name.find("emb") != std::string::npos) {
        return Memory::LayerTypes::EMB;
    }
    if (name.find("mha") != std::string::npos) {
        return Memory::LayerTypes::MHA;
    }
    if (name.find("ffn") != std::string::npos) {
        return Memory::LayerTypes::FFA;
    }
    std::ostringstream oss;
    oss << "Error: Layer type is unknown: " << name;
    throw std::logic_error(oss.str());
}

bool Memory::check_free_memory(
    const std::shared_ptr<Chakra::ETFeederNode> node) {
    uint64_t required_mem = 0;
    if (node->type() == ChakraNodeType::COMP_NODE) {
        for (auto attr : node->getChakraNode()->attr()) {
            if (attr.name() == "y_tensor_size") {
                required_mem = attr.int64_val() * this->get_num_bytes();
            }
        }
    } else {
        for (auto attr : node->getChakraNode()->attr()) {
            if (attr.name() == "comm_size") {
                required_mem = attr.int64_val() * this->get_num_bytes();
            }
        }
    }
    // Check for free memory.
    if (required_mem > this->get_free_memory()) {
        return false;
    } else {
        return true;
    }
}

void Memory::update_consumed_memory(
    const std::shared_ptr<Chakra::ETFeederNode> node, int sys_id) {

    this->is_oom = this->is_oom ? this->is_oom : !this->check_free_memory(node);

    // Get access to the Chakra node
    auto chakra_node = node->getChakraNode();

    auto child_nodes = node->getChildren();
    std::vector<uint64_t> children_ids;
    for (const auto& child : child_nodes) {
        children_ids.push_back(child->id());
    }

    auto node_id = node->id();

    // Check if we are in the forward or backward pass.
    this->is_forward_pass = !(node->name().find("_d") != std::string::npos);

    bool is_tensor = !(node->name().find("@0") != std::string::npos);
    bool is_embedding = node->name().find("emb") != std::string::npos;

    int other_mem_bytes = this->get_num_bytes();

    if (this->is_forward_pass) {
        if (is_tensor) {
            if (node->type() == ChakraNodeType::COMP_NODE && !is_embedding) {
                for (auto attr : chakra_node->attr()) {
                    if (attr.name() == "y_tensor_size") {
                        uint64_t tensor_size =
                            attr.int64_val() * other_mem_bytes;
                        this->max_parameter_memory += tensor_size;
                    }
                }
            }
            this->gradient_memory.removeChildIdFromAll(node_id);
        } else {
            if (!is_embedding &&
                (node->type() == ChakraNodeType::COMP_NODE ||
                 node->type() == ChakraNodeType::COMM_RECV_NODE)) {
                for (auto attr : chakra_node->attr()) {
                    if (attr.name() == "y_tensor_size") {
                        uint64_t tensor_size =
                            attr.int64_val() * other_mem_bytes;
                        this->activation_memory.addNode(node_id, tensor_size,
                                                        children_ids);
                        this->acumilated_activation_memory += tensor_size;
                    }
                }
            }
        }
        // Communication node
        // Replace current node ID in children lists with its actual
        // children
        this->activation_memory.replaceNodeWithChildrenEverywhere(node_id,
                                                                  children_ids);
    } else {  // Backward pass

        if (!is_tensor && !is_embedding &&
            (node->type() == ChakraNodeType::COMP_NODE ||
             node->type() == ChakraNodeType::COMM_RECV_NODE)) {
            for (auto attr : chakra_node->attr()) {
                if (attr.name() == "y_tensor_size") {
                    uint64_t tensor_size = attr.int64_val() * other_mem_bytes;
                    // this->gradient_memory.removeChildIdFromAll(node_id);
                    this->gradient_memory.addNode(node_id, tensor_size,
                                                  children_ids);
                    this->acumilated_gradient_memory += tensor_size;
                }
            }
        }
        // Remove node_id from activation_memory children lists
        this->activation_memory.removeChildIdFromAll(node_id);
        this->gradient_memory.replaceNodeWithChildrenEverywhere(node_id,
                                                                children_ids);
    }
    /*if (sys_id == 32) {
        for (auto [id, info] : this->gradient_memory.nodes_) {
            std::cout << "node id: " << node->id() << "," << id << " {";
            for (auto child : info.child_node_ids) {
                std::cout << child << ", ";
            }
            std::cout << "}" << std::endl;
        }
    }*/

    long long activation_mem = this->get_curr_activation_memory();
    long long gradient_mem = this->get_curr_gradient_memory();
    long long optimizer_mem = this->is_forward_pass && !is_tensor
                                  ? 0
                                  : this->get_max_optimizer_memory();
    this->curr_consumed_mem = this->get_max_parameter_memory() + optimizer_mem +
                              activation_mem + gradient_mem;

    this->get_max_activation_memory();
    this->get_max_gradient_memory();
    if (this->trace_mem) {
        LoggerFactory::get_memory_logger()->info(
            ",{}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}", sys_id,
            Sys::boostedTick(), node->id(), node->name(),
            static_cast<uint64_t>(node->type()), this->curr_consumed_mem,
            activation_mem, gradient_mem, this->get_max_parameter_memory(),
            optimizer_mem, this->get_is_oom());
    }
}

Memory::~Memory() {}
