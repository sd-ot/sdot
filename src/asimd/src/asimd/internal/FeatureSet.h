#pragma once

#include <string>

namespace asimd {

template<class... Features>
class FeatureSet {
    template<class... T> struct Has_;

    template<class F,class G,class... T> struct Has_<F,G,T...> { enum { value = Has_<F,T...>::value }; };
    template<class F,class... T> struct Has_<F,F,T...> { enum { value = true }; };
    template<class F> struct Has_<F> { enum { value = false }; };

    template<class... T> struct FeatureNames_ { static std::string get() { return ""; } };
    template<class H,class... T> struct FeatureNames_<H,T...> { static std::string get() { return "," + H::name() + FeatureNames_<T...>::get(); } };

public:
    template<class F>
    struct Has {
        enum { value = Has_<F,Features...>::value };
    };

    static std::string feature_names() {
        return FeatureNames_<Features...>::get();
    }
};


} //  namespace asimd
