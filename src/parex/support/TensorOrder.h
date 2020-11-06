#pragma once

#include <vector>
#include <array>
#include <set>

namespace parex {

template<class T>
struct TensorOrder {
    enum { value = 0 };
};

template<class T>
struct TensorOrder<std::vector<T>> {
    enum { value = 1 + TensorOrder<T>::value };
};

template<class T,std::size_t dim>
struct TensorOrder<std::array<T,dim>> {
    enum { value = 1 + TensorOrder<T>::value };
};

template<class T,class C,class A>
struct TensorOrder<std::set<T,C,A>> {
    enum { value = 1 + TensorOrder<T>::value };
};

}
