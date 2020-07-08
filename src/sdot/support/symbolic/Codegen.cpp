#include "Codegen.h"

namespace Symbolic {

Codegen::Codegen( std::string TF, std::string sp ) : TF( TF ), sp( sp ) {
    nb_regs = 0;
}

void Codegen::add_expr( std::string name, Expr expr ) {
    outputs.push_back( { name, expr } );
}

void Codegen::write( std::ostream &os ) {
    if ( outputs.empty() )
        return;

    ++outputs[ 0 ].expr.inst->context->date;
    for( const Output &output : outputs ) {
        write( os, output.expr.inst );
        if ( output.name == "return" )
            os << sp << "return " << output.expr.inst->reg << ";\n";
        else
            os << sp << TF << " " << output.name << " = " << output.expr.inst->reg << ";\n";
    }
}

void Codegen::write( std::ostream &os, Inst *inst ) {
    if ( inst->date == inst->context->date )
        return;
    inst->date = inst->context->date;

    for( Inst *ch : inst->children )
        write( os, ch );

    inst->reg = "R" + std::to_string( nb_regs++ );
    os << sp << TF << " " << inst->reg << " = ";
    inst->write_code( os );
    os << ";\n";
}

}
