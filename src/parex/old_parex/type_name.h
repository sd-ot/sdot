#pragma once

#include <string>
#include "S.h"

inline std::string type_name( S<std::string> ) { return "std::string"; }
inline std::string type_name( S<int        > ) { return "SI32";        }

template<class T> std::string type_name() { return type_name( S<T>() ); }
