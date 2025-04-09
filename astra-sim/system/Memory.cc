#include "astra-sim/system/Memory.hh"

#include "astra-sim/system/LogGP.hh"
#include "astra-sim/system/Sys.hh"

using namespace std;
using namespace AstraSim;

typedef ChakraProtoMsg::NodeType ChakraNodeType;
typedef ChakraProtoMsg::CollectiveCommType ChakraCollectiveCommType;

Memory::Memory() {
    this->memory_size = 0;
    this->consumed_memory = 0;
    this->parameter_memory = 0;
    this->optimizer_memory = 0;
    this->tmp_activation_memory = 0;
    this->tmp_gradient_memory = 0;
    this->is_forward_pass = true;
    this->layer_type_prev = Memory::LayerTypes::EMB;
    this->stop_recording_memory = false;
    this->prev_stack_number = 0;
}

void Memory::set_memory_size(long long size) {
    this->memory_size = size;
}

long long Memory::get_memory_size() {
    return this->memory_size;
}

long long Memory::get_consumed_memory() {
    this->consumed_memory =
        this->get_activation_memory() + this->get_gradient_memory() +
        this->get_parameter_memory() + this->get_optimizer_memory();

    return this->consumed_memory;
}

long long Memory::get_activation_memory() {
    return this->tmp_activation_memory;
}

long long Memory::get_gradient_memory() {
    return this->tmp_gradient_memory;
}

long long Memory::get_parameter_memory() {
    return this->parameter_memory;
}

long long Memory::get_optimizer_memory() {
    return this->parameter_memory * 2;  // In cased of mixed precision we do 3x
}

long long Memory::get_free_memory() {
    return this->memory_size - this->get_consumed_memory();
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

int Memory::get_curr_stack(std::string name) {
    std::regex pattern(R"(stack_(\d+)_)");
    std::smatch match;
    if (std::regex_search(name, match, pattern)) {
        return std::stoi(match[1]);
    }
    return -1;
}

/*
void Memory::update_consumed_memory(
    const std::shared_ptr<Chakra::ETFeederNode> node) {
    // Check for free memory.
    if (this->get_free_memory() + node->tensor_size() < 0) {
        std::ostringstream oss;
        oss << "Error: Reaching memory limits when executing the node: "
            << node->name() << " with Id " << node->id();
        throw std::logic_error(oss.str());
    }

    // Get access to the Chakra node
    auto chakra_node = node->getChakraNode();
    auto parent_nodes = node->getParents();
    auto child_nodes = node->getChildren();
    auto node_id = node->id();

    // Check if we are in the forward or backward pass.
    this->is_forward_pass = !(node->name().find("d_") != std::string::npos ||
                              node->name().find("_d") != std::string::npos);

    bool is_tensor = !(node->name().find("@0") != std::string::npos);

    // Get the parents
    // std::cout << "parents: "<< node->getParents().size() ;
    // for (auto parent : node->getParents())
    // std::cout << " node id: " << node->id() << " deps: " << parent->id() <<
    // ", "  << std::endl;

    if (!is_tensor) {  // Not a tensor
        if (this->is_forward_pass) {
            if (node->type() == ChakraNodeType::COMP_NODE ||
                node->type() == ChakraNodeType::COMM_RECV_NODE) {
                for (auto attr : chakra_node->attr()) {
                    if (attr.name() == "y_tensor_size") {
                        // TODO
                        // Add the activation to the activation_memory
                        // Add the children deps
                        // this->tmp_activation_memory += attr.int64_val();
                    }
                }
            } else {  // Communication node
                // TODO
                // Remove: look for the current node ID in the children deps of
                // the activations. Replace the current node id with the
                // children of this node Ex: [70-comp has child 90]
                //     [90-comm has child 89]
                //     [Replace the child 90 with the child 89]
                //     If not found, just ignore.
            }

        } else {  // Backward pass
            if (node->type() == ChakraNodeType::COMP_NODE ||
                node->type() == ChakraNodeType::COMM_RECV_NODE) {

                // TODO
                // Add the gradient to the gradient_memory
                // Add the children deps
                // Look for the current node ID in the activations childrens
                // Remove the matching ids from children lists.
                // this->tmp_gradient_memory += node->tensor_size();
            } else {  // Communication node
                // TODO
                // Remove: look for the current node ID in the children deps of
                // the gradients. Replace the current node id with the children
                // of this node Ex: [70-comp has child 90]
                //     [90-comm has child 89]
                //     [Replace the child 90 with the child 89]
                //     If not found, just ignore.
            }
        }

        this->layer_type_prev = this->str_to_layer_type(node->name());
    } else {  // Is a tensor node
        this->parameter_memory += node->tensor_size();
    }
}*/

void Memory::update_consumed_memory(
    const std::shared_ptr<Chakra::ETFeederNode> node, int sys_id) {
    // Check for free memory.
    if (this->get_free_memory() + node->tensor_size() < 0) {
        std::ostringstream oss;
        oss << "Error: Reaching memory limits when executing the node: "
            << node->name() << " with Id " << node->id();
        throw std::logic_error(oss.str());
    }

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

    if (this->is_forward_pass) {
        if (is_tensor) {
            this->parameter_memory += node->tensor_size();
            this->gradient_memory.removeChildIdFromAll(node_id);
        } else {
            if (node->type() == ChakraNodeType::COMP_NODE ||
                node->type() == ChakraNodeType::COMM_RECV_NODE) {
                for (auto attr : chakra_node->attr()) {
                    if (attr.name() == "y_tensor_size") {
                        uint64_t tensor_size = attr.int64_val();
                        this->activation_memory.addNode(node_id, tensor_size,
                                                        children_ids);
                        this->tmp_activation_memory += tensor_size;
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

        if (is_tensor) {
            this->parameter_memory += node->tensor_size();
        } else {
            if (node->type() == ChakraNodeType::COMP_NODE ||
                node->type() == ChakraNodeType::COMM_RECV_NODE) {
                uint64_t tensor_size = node->tensor_size();
                this->gradient_memory.addNode(node_id, tensor_size,
                                              children_ids);

                this->tmp_gradient_memory += tensor_size;

            }
        }
        // Remove node_id from activation_memory children lists
        this->activation_memory.removeChildIdFromAll(node_id);
        this->gradient_memory.replaceNodeWithChildrenEverywhere(node_id,
                                                                children_ids);
    }
    long long activation_mem = this->activation_memory.totalSize();
    long long gradient_mem = this->gradient_memory.totalSize();
    long long optimizer_mem =
        this->is_forward_pass && !is_tensor ? 0 : this->get_optimizer_memory();
    long long consumed_mem = this->get_parameter_memory() + optimizer_mem +
                             activation_mem + gradient_mem;
    LoggerFactory::get_logger("memory")->info(
        "memory, {}, {}, {}, {}, {}, memory {}, activation {}, gradient {}, "
        "parameter {}, optimizer "
        "{}.",
        sys_id, Sys::boostedTick(), node->id(), node->name(),
        static_cast<uint64_t>(node->type()), consumed_mem, activation_mem,
        gradient_mem, this->get_parameter_memory(), optimizer_mem);
}

Memory::~Memory() {}
