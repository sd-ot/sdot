#pragma once

#include <vector>
#include <string>

/**
*/
class GenIO {
public:
    virtual void write_field( const std::string &name, const double *data, const std::vector<std::size_t> &sizes ) = 0;
};
