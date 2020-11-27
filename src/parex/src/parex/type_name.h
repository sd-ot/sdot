#pragma once

#include <string>
#include "S.h"

inline std::string type_name( S<std::string> ) { return "std::string"; }

inline std::string type_name( S<double     > ) { return "FP64";        }
inline std::string type_name( S<float      > ) { return "FP32";        }

inline std::string type_name( S<int        > ) { return "SI32";        }

template<class T> std::string type_name( S<std::allocator<T>> ) { return "std::allocator<" + type_name( S<T>() ) + ">"; }

template<class T> std::string type_name() { return type_name( S<T>() ); }
