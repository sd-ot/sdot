#pragma once

template<int n,int v=1,int e=0,bool done=( n <= v )>
struct NextPow2;

template<int n,int v,int e>
struct NextPow2<n,v,e,true> {
    enum { value = v, expo = e };
};

template<int n,int v,int e>
struct NextPow2<n,v,e,false> {
    using N = NextPow2<n,v*2,e+1>;
    enum { value = N::value, expo = N::expo };
};

template<class TI>
TI next_pow2( TI val, TI *expo = nullptr ) {
    TI res = 1, e = 0;
    while ( res < val ) {
        res *= 2;
        ++e;
    }
    if ( expo )
        *expo = e;
    return res;
}
