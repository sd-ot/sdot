#pragma once

#include <vector>
#include <string>
#include <map>

class CompilationEnvironment;

namespace parex {
namespace hardware_information {
class ProcessingUnit;

/**/
class Memory {
public:
    struct              PULink          { ProcessingUnit *processing_unit; double bandwidth; };
    using               BwToPULink      = std::map<double,std::vector<PULink>>;
    using               PUToPULink      = std::map<ProcessingUnit *,PULink>;

    /**/                Memory          ();
    virtual            ~Memory          ();

    virtual void        write_to_stream ( std::ostream &os ) const = 0;
    virtual std::string allocator_type  () const = 0;
    virtual void*       allocator_data  () const = 0;

    void                register_link   ( const PULink &link );

    BwToPULink          bw_to_pu_links; ///< bandwith => processing unit with link info
    PUToPULink          pu_to_pu_link;  ///< processing unit => link info
    bool                is_local;       ///<
};

} // namespace hardware_information
} // namespace parex
