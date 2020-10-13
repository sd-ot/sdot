#include <algorithm>
#include <map>

#include "../support/ERROR.h"
#include "../support/P.h"
#include "KernelSlot.h"
#include "HwType.h"
#include "HwInfo.h"

namespace sdot {

KernelSlot::KernelSlot( std::string slot_name ) : slot_name( slot_name ) {
}

KernelSlot::~KernelSlot() {
}

// =====================================================================
KernelSlot::VK KernelSlot::available_slots( std::string TF, std::string TI ) {
    HwInfo hw_info;

    std::map<std::string,std::unique_ptr<KernelSlot>> best_kernels; // for each slots
    for( HwType *ht = HwType::last; ht; ht = ht->prev ) {
        VK available;
        ht->get_available_slots( available, hw_info, TF, TI );

        for( std::unique_ptr<KernelSlot> &ks : available )
            if ( best_kernels.count( ks->slot_name ) == 0 || best_kernels[ ks->slot_name ]->score() < ks->score() )
                best_kernels[ ks->slot_name ] = std::move( ks );
    }

    VK res;
    for( auto &sk : best_kernels )
        res.push_back( std::move( sk.second ) );

    if ( res.empty() )
        ERROR( "there's no kernel set for this set of parameters" );

    std::sort( res.begin(), res.end(), []( const std::unique_ptr<KernelSlot> &a, const std::unique_ptr<KernelSlot> &b ) {
        return a->score() > b->score();
    } );

    return res;
}

} // namespace sdot
