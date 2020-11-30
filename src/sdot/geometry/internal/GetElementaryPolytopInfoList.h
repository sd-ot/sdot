#pragma once

#include <parex/ComputableTask.h>
#include <parex/Value.h>

namespace sdot {

/**
*/
class GetElementaryPolytopInfoList : public ComputableTask {
public:
    /***/               GetElementaryPolytopInfoList( const Rc<Task> &shape_types );
    virtual void        write_to_stream             ( std::ostream &os ) const override;
    virtual void        exec                        () override;

private:
    static std::string  default_shape_types( int dim );
};

} // namespace sdot
