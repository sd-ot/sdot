#pragma once

#include "../internal/S.h"
#include "../internal/N.h"
#include <utility>
#include <string>

namespace asimd {
namespace processing_units {

/**
*/
template<class... Features>
class FeatureSet {
    template<class Feature,class T,class dummy=N<0>> struct SimdSizeFeature {
        static constexpr int value = 1;
    };
    template<class Feature,class T> struct SimdSizeFeature<Feature,T,N<Feature::template SimdSize<T>::value*0>> {
        static constexpr int value = Feature::template SimdSize<T>::value;
    };

    template<class... _Features> struct Content {
        template<class Feature> struct Has { enum { value = false }; };
        template<class T> struct SimdSize { static constexpr int value = 1; };
        static std::string feature_names( std::string = "," ) { return ""; }
    };

    template<class Head,class... Tail> struct Content<Head,Tail...> {
        using Next = Content<Tail...>;

        template<class Feature,int dummy=0> struct Has { enum { value = Next::template Has<Feature>::value }; };
        template<int dummy> struct Has<Head,dummy> { enum { value = true }; };

        template<class T> struct SimdSize { static constexpr int value = std::max( SimdSizeFeature<Head,T>::value, Next::template SimdSize<T>::value ); };

        static std::string feature_names( std::string prefix = "," ) { return prefix + Head::name() + Next::feature_names(); }

        template<class T> auto &value_( S<T> s ) { return next.value_( s ); }
        auto &value_( S<Head> ) { return value; }

        Head value;
        Next next;
    };

    using C = Content<Features...>;
    C content;

public:    
    template<class F>
    struct Has {
        enum { value = C::template Has<F>::value };
    };

    template<class T>
    struct SimdSize {
        enum { value = C::template SimdSize<T>::value };
    };

    static std::string feature_names() {
        return C::feature_names();
    }

    template<class T>
    auto &value() {
        return content.value_( S<T>() );
    }
};

} // namespace processing_units
} //  namespace asimd
