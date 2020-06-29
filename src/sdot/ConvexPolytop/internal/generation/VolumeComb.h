#pragma once

#include <functional>
#include "Volume.h"

/**
*/
class VolumeComb {
public:
    using               TI                = Volume::TI;
    using               Pt                = Volume::Pt;
    using               TF                = Volume::TF;

    /**/                VolumeComb        ( std::vector<Pt> pts, std::vector<std::string> allowed_volume_types );

    void                for_each_comb     ( const std::function<void( const std::vector<TI> &inds )> &f );

    std::vector<Volume> possible_volumes; ///<
    std::vector<bool>   are_disjoint;     ///<

private:
    void                for_each_comb_    ( const std::function<void(const std::vector<TI> &)> &f, std::vector<bool> possible_set, std::vector<TI> inds, TI ind );
};
