#pragma once

#include <atomic>
#include <vector>
#include <string>
#include <map>

class CompilationEnvironment;

namespace parex {
class ProcessingUnit;

/**/
class Memory {
public:
    struct              PULink          { ProcessingUnit *processing_unit; double bandwidth; };
    using               BwToPULink      = std::map<double,std::vector<PULink>>; ///< bandwidth => links to processing units
    using               PUToPULink      = std::map<ProcessingUnit *,PULink>; ///<processing units => links to processing units
    using               I               = std::uint64_t;

    /**/                Memory          ();
    virtual            ~Memory          ();

    virtual void        write_to_stream ( std::ostream &os ) const = 0;
    virtual std::string allocator_type  () const = 0;
    virtual void*       allocator_data  () = 0;

    void                register_link   ( const PULink &link );

    BwToPULink          bw_to_pu_links; ///< bandwith => processing unit with link info
    PUToPULink          pu_to_pu_link;  ///< processing unit => link info
    I                   amount;         ///< in bytes
    std::atomic<I>      used;           ///< in bytes
};

} // namespace parex
