#pragma once

#include <memory>
#include <vector>
#include <string>

namespace sdot {
class KernelSlot;
class HwInfo;

/**
*/
class HwType {
public:
    /**/           HwType             ( std::string TF_name, std::string TI_name );

    virtual void   get_available_slots( std::vector<std::unique_ptr<KernelSlot>> &available, const HwInfo &hw_info, std::string TF_name, std::string TI_name ) = 0; ///< get [slot_name,score] for each slot

    std::string    TF_name;           ///<
    std::string    TI_name;           ///<
    static HwType* last;              ///<
    HwType*        prev;              ///<
};

}
