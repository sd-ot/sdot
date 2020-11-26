#pragma once

#include <ostream>
#include <utility>
#include <memory>
#include <string>
#include <tuple>

template<class T>
struct HasWriteToStream {
    template<class    U> static auto has_write_to_stream( const U& val, std::ostream &os ) -> typename std::tuple_element<1,std::tuple<decltype(val.write_to_stream(os)),std::true_type>>::type { return {}; }
    template<class ...U> static auto has_write_to_stream( const U& ... ) -> std::false_type { return {}; }
    using                            OutType            = decltype( has_write_to_stream( *reinterpret_cast<T *>( 0 ), *reinterpret_cast<std::ostream *>( 0 ) ) );
    enum {                           value              = OutType::value };
};

template<class T>
struct HasBegin {
    template<class    U> static auto has_begin( const U& val ) -> typename std::tuple_element<1,std::tuple<decltype(val.begin()),std::true_type>>::type { return {}; }
    template<class ...U> static auto has_begin( const U& ... ) -> std::false_type { return {}; }
    using                            OutType  = decltype( has_begin( *reinterpret_cast<T *>( 0 ) ) );
    enum {                           value    = OutType::value };
};

template<class T>
struct HasPop {
    // template<class    U> static auto has_pop ( U val ) -> typename std::tuple_element<1,std::tuple<decltype(val.pop()),std::true_type>>::type { return {}; }
    template<class    U> static auto has_pop ( U& val ) -> typename std::tuple_element<1,std::tuple<decltype(val.pop()),std::true_type>>::type { return {}; }
    template<class ...U> static auto has_pop ( U& ... ) -> std::false_type { return {}; }
    using                            OutType = decltype( has_pop( *reinterpret_cast<T *>( 0 ) ) );
    enum {                           value   = OutType::value };
};

// operator<<( ostream ) if we have a `write to stream`
template<class T>
auto operator<<( std::ostream &os, const T &val ) -> typename std::enable_if<HasWriteToStream<T>::value,std::ostream &>::type {
    val.write_to_stream( os );
    return os;
}

// operator<<( ostream ) if we have a `begin` but no `write to stream`
template<class T>
auto operator<<( std::ostream &os, const T &val ) -> typename std::enable_if<HasBegin<T>::value && ! HasWriteToStream<T>::value,std::ostream &>::type {
    size_t s = 0;
    for( const auto &v : val )
        os << ( s++ ? "," : "" ) << v;
    return os;
}

// operator<<( ostream ) if we have a `pop` but no `write to stream`
template<class T>
auto operator<<( std::ostream &os, const T &val ) -> typename std::enable_if<HasPop<T>::value && ! HasWriteToStream<T>::value,std::ostream &>::type {
    os << '[';
    int cpt = 0;
    for( T cp = val; cp; ++cpt )
        os << ( cpt ? "," : "" ) << cp.pop();
    os << ']';
    return os;
}

//
template<class T>
std::ostream &operator<<( std::ostream &os, const std::unique_ptr<T> &val ) {
    if ( val )
        os << *val;
    else
        os << "NULL";
    return os;
}

// needed to avoid an ambiguous overload
inline std::ostream &operator<<( std::ostream &os, const std::string &val ) {
    os.write( val.data(), val.size() );
    return os;
}

//
template<class A,class B>
std::ostream &operator<<( std::ostream &os, const std::pair<A,B> &val ) {
    return os << "[" << val.first << "," << val.second << "]";
}
