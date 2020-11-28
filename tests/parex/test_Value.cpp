#include <parex/Scheduler.h>
#include <parex/Value.h>
#include <parex/P.h>

// using namespace parex;

int main() {
    // scheduler.log = true;

    P( Value( 16 ) );
    P( Value( 16 ) + Value( 17 ) );
}
