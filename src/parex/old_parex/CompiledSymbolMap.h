#pragma once

#include <dynalo/dynalo.hpp>
#include "SrcWriter.h"
#include <memory>
class TmpDir;

/**
*/
class CompiledSymbolMap {
public:
    using                     Path               = SrcWriter::Path;

    template<class TF> TF*    symbol_for         ( const std::string &parameters ) { return reinterpret_cast<TF *>( untyped_symbol_for( parameters ) ) ;}

protected:
    friend class              SrcWriter;

    virtual Path              output_directory   ( const std::string &parameters ) const = 0;
    virtual void              make_srcs          ( SrcWriter &cff ) const = 0;

    virtual void              make_cmakelists    ( std::ostream &os, const SrcWriter &ff, const std::string &shash ) const;
    virtual std::string       symbol_name        ( const std::string &parameters ) const;
    virtual int               exec               ( const std::string &cmd ) const;

private:
    struct                    DF                 { std::unique_ptr<dynalo::library> lib; void *sym; };

    void*                     untyped_symbol_for ( const std::string &parameters );
    DF                        load_or_make_lib   ( const std::string &parameters );
    DF                        load_sym           ( const Path &output_directory, const std::string &shash, const std::string &symbol_name );
    DF                        make_lib           ( const Path &output_directory, const Path &pinfo, const std::string &shash, const std::string &parameters, const std::string &symbol_name );

    std::map<std::string,DF>  df_map;
};
