#include "KernelSlot_Cpu_Gen.h"
#include "HwType.h"
#include "HwInfo.h"

namespace sdot {

HwType *HwType::last = nullptr;

HwType::HwType( std::string TF_name, std::string TI_name ) : TF_name( TF_name ), TI_name( TI_name ), prev( last ) {
    last = this;
}

// =====================================================================
template<class TF,class TI>
class HwForKernelsType_Cpu_Gen : public HwType {
public:
    /**/          HwForKernelsType_Cpu_Gen( std::string TF_name, std::string TI_name ) : HwType( TF_name, TI_name ) {}

    virtual void  get_available_slots     ( std::vector<std::unique_ptr<KernelSlot>> &available, const HwInfo &/*hw_info*/, std::string TF_name, std::string TI_name ) {
        if ( this->TF_name != TF_name || this->TI_name != TI_name )
            return;
        available.push_back( std::make_unique<Kernels_Cpu_Gen<TF,TI>>( "cpu 0" ) );
    }
};

static HwForKernelsType_Cpu_Gen<double,std::uint64_t> inst_HwForKernelsType_Cpu_Gen_double_uint64_t( "double", "uint64" );

}
