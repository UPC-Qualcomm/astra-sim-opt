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
    return this->parameter_memory * 3;
}

long long Memory::get_free_memory() {
    return this->memory_size - this->get_consumed_memory();
}

void Memory::release_activation() {
    // Remove the activations of the layers that we have already calculated
    // their gradients.
}

void Memory::release_gradient() {
    // Remove the gradients after the parameters of the model are updated Or the
    // bwd pass is finished.
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

void Memory::update_consumed_memory(
    const std::shared_ptr<Chakra::ETFeederNode> node) {
    if (this->get_free_memory() + node->tensor_size() < 0) {
        std::ostringstream oss;
        oss << "Error: Reaching memory limits when executing the node: "
            << node->name() << " with Id " << node->id();
        throw std::logic_error(oss.str());
    }
    int stack_number = -1;
    if (node->name().find("stack") != std::string::npos) {
        stack_number = this->get_curr_stack(node->name());
        if (stack_number < this->prev_stack_number) {
            this->stop_recording_memory = true;
        } else {
            this->prev_stack_number = stack_number;
        }
    }
    bool is_backward = node->name().find("d_") != std::string::npos;
    if (is_backward) {
        this->is_forward_pass = false;
    } else if (node->type() == ChakraNodeType::COMP_NODE) {
        this->is_forward_pass = true;
    }
    if (!this->stop_recording_memory) {
        if (node->type() == ChakraNodeType::COMP_NODE) {
            if (this->is_forward_pass) {
                for (auto attr : node->getChakraNode()->attr()) {
                    if (attr.name() == "y_tensor_size") {
                        this->tmp_activation_memory += attr.int64_val();
                    }
                }
            } else {
                this->tmp_gradient_memory += node->tensor_size();
                this->parameter_memory += node->tensor_size();
            }

            /*if (node->name().find("_w") != std::string::npos ||
                node->name().find("_dw") != std::string::npos ||
                node->name().find("emb_") != std::string::npos) {
                this->parameter_memory += node->tensor_size();
            }*/

            this->layer_type_prev = this->str_to_layer_type(node->name());
        } else if (node->type() == ChakraNodeType::COMM_COLL_NODE || // node->type() == ChakraNodeType::COMM_SEND_NODE (This will not affect the memory size)
                   node->type() == ChakraNodeType::COMM_RECV_NODE) {

            int64_t comm_size = 0;
            int64_t y_tensor_size = 0;


            comm_size = node->comm_size();
            for (auto attr : node->getChakraNode()->attr()) {
                if (attr.name() == "y_tensor_size") {
                    y_tensor_size += attr.int64_val();
                }
            }

            // TODO: Consider the special case of each communication type.
            /*if (node->comm_type() == ChakraCollectiveCommType::ALL_GATHER ||
                node->comm_type() == ChakraCollectiveCommType::GATHER ||
                node->comm_type() == ChakraCollectiveCommType::ALL_TO_ALL ||
                node->comm_type() == ChakraCollectiveCommType::ALL_REDUCE ||
                node->comm_type() == ChakraCollectiveCommType::REDUCE_SCATTER||
                node->comm_type() == ChakraCollectiveCommType::REDUCE){*/
                
            if (this->is_forward_pass) {
                this->tmp_activation_memory += y_tensor_size;
                //this->communication_memory += comm_size;
            } else {
                this->tmp_gradient_memory += y_tensor_size;
                this->parameter_memory += y_tensor_size;
                //this->communication_memory += comm_size;
            }
        //}
        }
    }
}

Memory::~Memory() {}
