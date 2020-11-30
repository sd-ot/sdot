#pragma once

#include <string>
#include <vector>

/**
*/
class ElementaryPolytopInfoListContent {
public:
    struct ElemInfo {
        std::string name;
    };

    std::vector<ElemInfo> elem_info;
};

