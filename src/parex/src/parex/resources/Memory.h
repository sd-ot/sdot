#pragma once

#include <atomic>
#include <vector>
#include <string>
#include <map>

class CompilationEnvironment;

namespace parex {
class Processor;

/**/
class Memory {
public:
    using               I               = std::uint64_t;
    struct              ProcLink        { Processor *processing_unit; double bandwidth; };
    using               BwToProcLink    = std::map<double,std::vector<ProcLink>>; ///< bandwidth => links to processing units
    using               ProcToProcLink  = std::map<Processor *,ProcLink>; ///<processing units => links to processing units

    /**/                Memory          ();
    virtual            ~Memory          () {}

    virtual void        write_to_stream ( std::ostream &os ) const = 0;
    virtual std::string allocator_type  () const = 0;
    virtual void*       allocator_data  () = 0;

    void                register_link   ( const ProcLink &link );

    BwToProcLink        bw_to_pu_links; ///< bandwith => processing unit with link info
    ProcToProcLink      pu_to_pu_link;  ///< processing unit => link info
    I                   amount;         ///< in bytes
    std::atomic<I>      used;           ///< in bytes
};

} // namespace parex
