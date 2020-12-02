#include <catch2/catch_test_macros.hpp>

/////
//template<class T,class M>
//struct WithinAbsMatcher : Catch::MatchExpr(cherBase<T> {
//    WithinAbsMatcher(T target, M margin)
//        :m_target{ target }, m_margin{ margin } {
//        CATCH_ENFORCE( margin >= 0, "Invalid margin: " << margin << '.'
//            << " Margin has to be non-negative.");
//    }
//    bool match(T const& matchee) const override {
//        return ( matchee + m_margin >= m_target ) && ( m_target + m_margin >= matchee );
//    }
//    std::string describe() const override {
//        return "is within " + Catch::Detail::stringify( m_margin ) + " of " + Catch::Detail::stringify( m_target );
//    }
//private:
//    T m_target;
//    M m_margin;
//};

//template<class T,class M>
//WithinAbsMatcher<T,M> WithinAbs( T target, M margin ) {
//    return WithinAbsMatcher<T,M>( target, margin );
//}

