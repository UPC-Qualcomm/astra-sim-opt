#ifndef __MEMORY_HH__
#define __MEMORY_HH__

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include<bits/stdc++.h> 
#include <algorithm> 
#include <regex>

#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Common.hh"
#include "extern/graph_frontend/chakra/src/feeder/et_feeder.h"

namespace AstraSim {

class Memory {
  public:

    enum LayerTypes : int {
      EMB,
      MHA,
      FFA
    };
    
    Memory();
    void set_memory_size(long long size);
    long long get_memory_size();
    long long get_consumed_memory();
    long long get_activation_memory();
    long long get_gradient_memory();
    long long get_parameter_memory();
    long long get_optimizer_memory();
    long long get_free_memory();
    void release_activation();
    void release_gradient();

    int get_curr_stack(std::string node_name);
    std::string layer_type_to_str(LayerTypes type);
    Memory::LayerTypes str_to_layer_type(std::string node_name);
    void update_consumed_memory(
        const std::shared_ptr<Chakra::ETFeederNode> node);
    
    ~Memory();

    long long memory_size;
    long long consumed_memory;
    std::vector<long long> activation_memory;
    long long parameter_memory;
    std::vector<long long> gradient_memory;
    long long optimizer_memory;
    long long tmp_activation_memory;
    long long tmp_gradient_memory;
    bool is_forward_pass;
    int layer_type_prev;
    int stop_recording_memory;
    int prev_stack_number;

};

}  // namespace AstraSim

#endif /* __MEMORY_HH__ */
