#pragma once

#include "CompiledTask.h"

/**
*/
class CompiledTaskWithGeneratedSrc : public CompiledTask {
public:
    using               CompiledTask::CompiledTask;

    virtual std::string func_name() override;
};

