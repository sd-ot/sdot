#pragma once

#include "Type.h"

namespace parex {

/**
*/
class CppType : public Type {
public:

    /**/                 CppType           ( std::string name, const CompilationEnvironment &compilation_environment, std::vector<Type *> &&sub_types = {} );

};

} // namespace parex
