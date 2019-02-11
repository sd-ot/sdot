#pragma once


/**
*/
struct GetTail {
    template<class U>
    struct Trans {
        using T = typename U::Tail;
    };
};
