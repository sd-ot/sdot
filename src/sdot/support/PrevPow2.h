#pragma once

template<int n,int v=1,int e=0,bool done=( n < v )>
struct PrevPow2;

template<int n,int v,int e>
struct PrevPow2<n,v,e,true> {
    enum { value = v / 2, expo = e - 1 };
};

template<int n,int v,int e>
struct PrevPow2<n,v,e,false> {
    using N = PrevPow2<n,v*2,e+1>;
    enum { value = N::value, expo = N::expo };
};

