#pragma once

#include <filesystem>
#include "Task.h"
#include "Rc.h"

/**
*/
class ComputableTask : public Task {
public:
    using                 Path           = std::filesystem::path;

    /***/                 ComputableTask ( std::vector<Rc<Task>> &&children );
    virtual              ~ComputableTask ();

    bool                  all_ch_computed() const;

    virtual void          get_front_rec  ( std::map<int,std::vector<ComputableTask *>> &front ) override;
    virtual bool          is_computed    () const override;
    virtual Type*         output_type    () const override;
    virtual void*         output_data    () const override;

    virtual void          exec           () = 0;

    std::vector<Rc<Task>> children;      ///<

    bool                  scheduled;     ///<
    bool                  in_front;      ///<
    bool                  computed;      ///<
    int                   priority;      ///<
    Type*                 type;          ///<
    void*                 data;          ///<
};

