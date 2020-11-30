#include <parex/ComputableTask.h>
#include <parex/Scheduler.h>
#include <parex/Value.h>
#include <parex/P.h>

// using namespace parex;

int main() {
    P( Value( 16 ) );
    P( Value( 16 ) + Value( 17 ) );

    Rc<Task> t = Value( 16 ).to_string();
    dynamic_cast<ComputableTask *>( t.data )->exec();
    P( *reinterpret_cast<std::string *>( t->output_data ) );
    P( *t->output_type );

    Rc<Task> u = Value( 16 ).conv_to<std::size_t>();
    dynamic_cast<ComputableTask *>( u.data )->exec();
    P( *reinterpret_cast<std::size_t *>( u->output_data ) );
    P( *u->output_type );
}
