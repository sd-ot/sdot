/*
 Copyright 2012 Hugo Leclerc

 This file is part of Hpipe.

 Hpipe is free software: you can redistribute it and/or modify
 it under the terms of the GNU Lesser General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 Hpipe is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU Lesser General Public License for more details.
 You should have received a copy of the GNU General Public License
 along with Hpipe. If not, see <http://www.gnu.org/licenses/>.
*/


#include "DotOut.h"
#include <string.h>
#include <stdlib.h>
#include <iostream>

std::ostream &dot_out( std::ostream &os, const char *beg, int lim ) {
    return dot_out( os, beg, beg + strlen( beg ), lim );
}

std::ostream &dot_out( std::ostream &os, const char *beg, const char *end, int lim ) {
    while ( beg < end && ( *beg == ' ' || *beg == '\n' ) )
        ++beg;

    for( ; beg < end; ++beg, --lim ) {
        if ( lim == 3 and end - beg > 3 ) {
            os << "...";
            break;
        }

        if ( *beg == '"' )
            os << "&quot;";
        else if ( *beg == '\n' )
            os << '\\' << 'n';
        else if ( *beg == '&' )
            os << "&amp;";
        else
            os << *beg;
    }

    return os;
}

int exec_dot( const std::string &f, const char *viewer, bool launch_viewer, bool par ) {
    // call graphviz
    std::ostringstream ss;
    ss << "dot -Tpdf " << f << " > " << f << ".pdf";
    // PRINT( ss.str() );
    if ( int res = system( ss.str().c_str() ) )
        return res;

    // call viewer
    if ( launch_viewer ) {
        const char *t[] = {
            viewer,
            "okular",
            "gv -widgetless",
            "xpdf",
            "acroread",
            "evince",
            0,
        };
        for( int i = not viewer; t[ i ]; ++i ) {
            std::ostringstream ss;
            ss << t[ i ] << " " << f << ".pdf";
            if ( par )
                ss << " 2> /dev/null &";
            if ( system( ss.str().c_str() ) == 0 )
                return 0;
        }
        std::cerr << "Impossible to find a pdf viewer" << std::endl;
    }

    return 0;
}

