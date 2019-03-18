#pragma once

/**
*/
struct TypeTransId {
    template<class U>
    struct Trans {
        using T = U;
    };
};
