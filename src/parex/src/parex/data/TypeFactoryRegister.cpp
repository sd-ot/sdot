#include "TypeFactoryRegister.h"

namespace parex {

TypeFactoryRegister *last_type_factory_registrar = nullptr;

TypeFactoryRegister::TypeFactoryRegister( std::vector<std::string> names, FuncVariant &&f ) : func_variant( std::move( f ) ), names( names ) {
    prev_type_factory_registrar = last_type_factory_registrar;
    last_type_factory_registrar = this;
}

void TypeFactoryRegister::reg( TypeFactory &tf ) {
    struct Reg {
        void operator()( std::function<Type*(const std::string &name, const std::string &base_name, TypeFactory &tf, const std::vector<std::string> &parameters)> &&f ) {
            for( std::string base_name : names ) {
                tf.reg_template( base_name, [ f{ std::move( f ) } ]( const std::string &name, const std::string &base_name, TypeFactory &tf, const std::vector<std::string> &parameters ) {
                    return f( name, base_name, tf, parameters );
                } );
            }
        }
        void operator()( std::function<Type*(TypeFactory &tf, const std::vector<std::string> &parameters)> &&f ) {
            for( std::string base_name : names ) {
                tf.reg_template( base_name, [ f{ std::move( f ) } ]( const std::string &/*name*/, const std::string &/*base_name*/, TypeFactory &tf, const std::vector<std::string> &parameters ) {
                    return f( tf, parameters );
                } );
            }
        }
        void operator()( const std::function<Type*(const std::string &name)> &f ) {
            for( std::string name : names )
                tf.reg_type( name, f );
        }
        void operator()( const std::function<void(Type*)> &f ) {
            for( std::string name : names )
                tf.reg_type( name, f );
        }

        std::vector<std::string> names;
        TypeFactory &tf;
    };
    std::visit( Reg{ names, tf }, std::move( func_variant ) );
}

} // namespace parex
