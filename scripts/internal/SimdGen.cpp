#include "../../src/sdot/support/ASSERT.h"
#include "SimdGen.h"
#include <ostream>

void SimdGen::add_write( std::string name, ST nop ) {
    writes.push_back( { .name = name, .nop = nop } );
}

SimdGen::ST SimdGen::new_undefined( SimdGen::ST len, std::string type ) {
    return new_op( Op::Type::Undefined, len, type );
}

SimdGen::ST SimdGen::new_gather( std::vector<ST> nops, std::vector<ST> nouts ) {
    // we already have it ?
    for( ST i = 0; i < ops.size(); ++i )
        if ( ops[ i ].type == Op::Type::Gather && ops[ i ].children == nops && ops[ i ].nouts == nouts )
            return i;

    // [ undefined, undefined, ... ]
    for( std::size_t i = 0; ; ++i ) {
        if ( i == nops.size() )
            return new_undefined( nops.size(), ops[ nops[ 0 ] ].scalar_type );
        if ( ops[ nops[ i ] ].type != Op::Type::Undefined )
            break;
    }

    // op[ 0, 1, 2, 3 ] when op.len == 3 => return op
    //    if ( nops.size() && nops.size() == ops[ nops[ 0 ] ].len ) {
    //        for( std::size_t i = 0; ; ++i ) {
    //            if ( i == nops.size() )
    //                return nops[ 0 ];
    //            if ( nops[ i ] != nops[ 0 ] || nouts[ i ] != i )
    //                break;
    //        }
    //    }

    //
    ST res = new_op( Op::Type::Gather, nops.size(), ops[ nops[ 0 ] ].scalar_type );
    ops[ res ].children = nops;
    ops[ res ].nouts = nouts;
    return res;
}

SimdGen::ST SimdGen::new_bop( SimdGenOp::Type type, ST a, ST b ) {
    //
    if ( Op::commutative( type ) && a > b )
        std::swap( a, b );

    // we already have it ?
    for( ST i = 0; i < ops.size(); ++i )
        if ( ops[ i ].type == type && ops[ i ].children[ 0 ] == a && ops[ i ].children[ 1 ] == b )
            return i;

    // else, make a new one
    ASSERT( ops[ a ].len == ops[ b ].len, "" );
    ST res = new_op( type, ops[ a ].len, ops[ a ].scalar_type );
    ops[ res ].children = { a, b };
    return res;
}

SimdGen::ST SimdGen::new_var( std::string name, ST len, std::string scalar_type ) {
    // we already have it ?
    for( ST i = 0; i < ops.size(); ++i ) {
        if ( ops[ i ].type == Op::Type::Variable && ops[ i ].name == name ) {
            ASSERT( ops[ i ].len == len && ops[ i ].scalar_type == scalar_type, "" );
            return i;
        }
    }
    // else, make a new one
    ST res = new_op( Op::Type::Variable, len, scalar_type );
    ops[ res ].name = name;
    return res;
}

SimdGen::ST SimdGen::new_op( SimdGenOp::Type type, ST len, std::string scalar_type ) {
    ST res = ops.size();
    ops.push_back( { type, len, scalar_type } );
    return res;
}

bool SimdGen::all_children_done( const std::vector<bool> &done, ST nop ) {
    for( ST ch : ops[ nop ].children )
        if ( ! done[ ch ] )
            return false;
    return true;
}

void SimdGen::handle_undefined() {
    // remove undefined writes
    for( ST num_wr = 0; num_wr < writes.size(); ++num_wr )
        if ( ops[ writes[ num_wr ].nop ].type == Op::Type::Undefined )
            writes.erase( writes.begin() + num_wr-- );

    //
}

void SimdGen::for_each_child( std::function<void( ST )> f, std::vector<bool> &seen, ST nop ) {
    if ( seen[ nop ] )
        return;
    seen[ nop ] = true;

    f( nop );

    for( ST ch : ops[ nop ].children )
        for_each_child( f, seen, ch );
}

void SimdGen::for_each_child( std::function<void( ST )> f ) {
    std::vector<bool> seen( ops.size(), false );
    for( const Write &wr : writes )
        for_each_child( f, seen, wr.nop );
}

void SimdGen::gen_code( std::ostream &os, std::string sp ) {
    // simplifications
    handle_undefined();

    //
    update_parents();

    std::vector<bool> done( ops.size(), false );
    std::vector<ST> front;
    for_each_child( [&]( ST nop ) {
        if ( ops[ nop ].children.empty() )
            front.push_back( nop );
    } );

    std::vector<std::string> tmps( ops.size() );
    while ( front.size() ) {
        ST nop = front.back();
        front.pop_back();

        if ( done[ nop ] )
            continue;
        done[ nop ] = true;

        write_inst( os, sp, tmps, nop );

        for( ST np : ops[ nop ].parents )
            if ( all_children_done( done, np ) )
                front.push_back( np );
    }

    for( const Write &wr : writes )
        if ( tmps[ wr.nop ] != wr.name )
            os << sp << wr.name << " = " << tmps[ wr.nop ] << ";\n";
}

void SimdGen::write_inst( std::ostream &os, std::string sp, std::vector<std::string> &tmps, ST nop ) {
    // variable
    const Op &op = ops[ nop ];
    if ( op.type == Op::Type::Variable ) {
        tmps[ nop ] = op.name;
        return;
    }

    // undefined
    if ( op.type == Op::Type::Undefined ) {
        tmps[ nop ] = "undefined";
        return;
    }

    // => we will make a new variable
    std::size_t n = 0;
    for( const std::string &tmp : tmps )
        n += ! tmp.empty();
    std::string tmp = "t" + std::to_string( n );
    tmps[ nop ] = tmp;

    //
    if ( op.type == Op::Type::Gather ) {
        //
        bool has_undefined = false;
        for( std::size_t i = 0; i < op.children.size(); ++i )
            if ( ops[ op.children[ i ] ].type == Op::Type::Undefined )
                has_undefined = true;

        if ( has_undefined ) {
            os << sp << "SimdVec<" << op.scalar_type << "," << op.len << "> " << tmp << ";\n";
            for( std::size_t i = 0; i < op.children.size(); ++i )
                if ( ops[ op.children[ i ] ].type != Op::Type::Undefined )
                    os << sp << tmp << "[ " << i << " ] = " << tmps[ op.children[ i ] ] << "[ " << op.nouts[ i ] << " ];\n";
        } else {
            os << sp << "SimdVec<" << op.scalar_type << "," << op.len << "> " << tmp << "{";
            for( std::size_t i = 0; i < op.children.size(); ++i )
                os << ( i ? ", " : " " ) << tmps[ op.children[ i ] ] << "[ " << op.nouts[ i ] << " ]";
            os << " };\n";
        }

        return;
    }

    // => binary op
    os << sp << "SimdVec<" << op.scalar_type << "," << op.len << "> " << tmp << " = "
       << tmps[ op.children[ 0 ] ] << " " << op.str_op() << " " << tmps[ op.children[ 1 ] ] << ";\n";

}

void SimdGen::update_parents() {
    for_each_child( [&]( ST nop ) {
        ops[ nop ].parents.clear();
    } );

    for_each_child( [&]( ST nop ) {
        for( ST ch : ops[ nop ].children )
            ops[ ch ].parents.push_back( nop );
    } );
}
