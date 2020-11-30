#pragma once

#include <functional>
#include <ostream>
#include <memory>
#include <string>
#include <vector>
class Src;

/**
*/
class Type {
public:
    using               UPType                    = std::unique_ptr<Type>;

    virtual            ~Type                      ();

    virtual void        for_each_include_directory( const std::function<void(std::string)> &cb ) const;
    virtual UPType      copy_with_sub_type        ( std::string name, std::vector<Type *> &&sub_types ) const = 0;
    virtual void        for_each_include          ( const std::function<void(std::string)> &cb ) const;
    virtual void        for_each_prelim           ( const std::function<void(std::string)> &cb ) const;
    virtual void        write_to_stream           ( std::ostream &os ) const = 0;
    virtual std::string cpp_name                  () const = 0;
    virtual void        destroy                   ( void *data ) const = 0;

    void                add_needs_in              ( Src &src ) const;
};

