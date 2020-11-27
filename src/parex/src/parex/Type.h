#pragma once

#include <functional>
#include <ostream>
#include <memory>
#include <string>

/**
*/
class Type {
public:
    using               UPType            = std::unique_ptr<Type>;

    virtual            ~Type              ();

    virtual UPType      copy_with_sub_type( std::vector<Type *> &&sub_types ) const = 0;
    virtual void        for_each_prelim   ( const std::function<void(std::string)> &cb ) const;
    virtual void        for_each_include  ( const std::function<void(std::string)> &cb ) const;
    virtual void        write_to_stream   ( std::ostream &os ) const = 0;
    virtual std::string cpp_name          () const = 0;
    virtual void        destroy           ( void *data ) const = 0;
};

