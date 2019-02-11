#pragma once

void abort_or_throw();
bool __disp_and_abort_if_not_cond__( bool cond, const char *txt, ... );
bool __disp( const char *txt, ... );
bool __do_nothing__();
#ifdef DEBUG
    #define ASSERT_IF_DEBUG( A ) __disp_and_abort_if_not_cond__( A, "%s:%i: assertion %s not checked\n", __FILE__, __LINE__, #A )
#else
    #define ASSERT_IF_DEBUG( A ) __do_nothing__()
#endif // DEBUG

#define ASSERT( A, txt, ... ) __disp_and_abort_if_not_cond__( A, "%s:%i: assertion %s not checked -> " txt "\n", __FILE__, __LINE__, #A, ##__VA_ARGS__ )

#define ERROR( txt, ... ) __disp_and_abort_if_not_cond__( 0, "%s:%i: " txt "\n", __FILE__, __LINE__, ##__VA_ARGS__ )

#define TODO ASSERT( 0, "TODO" )

#define WARNING( A, ... ) __disp( "%s:%i: " #A "\n", __FILE__, __LINE__, ##__VA_ARGS__ )
