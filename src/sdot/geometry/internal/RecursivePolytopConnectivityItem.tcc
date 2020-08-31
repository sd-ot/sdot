#include "RecursivePolytopConnectivityItem.h"

// write_to_stream -----------------------------------------------------------------------
template<class TF,class TI,int nvi>
void RecursivePolytopConnectivityItem<TF,TI,nvi>::write_to_stream( std::ostream &os ) const {
    os << "[";
    for( TI i = 0; i < faces.size(); ++i )
        faces[ i ].write_to_stream( os << ( i++ ? "," : "" ) );
    os << "]";
}

template<class TF,class TI>
void RecursivePolytopConnectivityItem<TF,TI,0>::write_to_stream( std::ostream &os ) const {
    os << node_number;
}

