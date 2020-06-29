#pragma once

#include <bitset>
#include <limits>

template<std::size_t N1,std::size_t N2>
typename std::enable_if<( N1 + N2 ) <= ( std::numeric_limits<unsigned long long>::digits ), std::bitset<N1 + N2> >::type
cat( const std::bitset<N1>& a, const std::bitset<N2>& b ) { return ( a.to_ullong() << N2 ) + b.to_ullong() ; }

template<std::size_t N1,std::size_t N2>
typename std::enable_if<( ( N1 + N2 ) > ( std::numeric_limits<unsigned long long>::digits ) ), std::bitset<N1+N2> >::type
cat( const std::bitset<N1>& a, const std::bitset<N2>& b ) { return std::bitset<N1+N2>( a.to_string() + b.to_string() ) ; }
