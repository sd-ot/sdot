#include "../src/sdot/ParEx/Scheduler.h"
#include "../src/sdot/ParEx/Kernel.h"
#include "../src/sdot/ParEx/Node.h"
#include "../src/sdot/support/P.h"

using namespace parex;
using TF = double;
using TI = int;

int main() {
    // Uniform[ "FP64" ]
    // MinMax[ dim ]
    // Display
    // MinMax[ dim ]
    //    int dim = 2;
    //    int nb_diracs = 200;
    Scheduler sch;
    //    Node p = Node::New( uniform( "FP64" ), 10 );
    //    sch << Node::New( display(), &std::cout );
    Node p = Node::New( "", 10 );
    //    sch << Node::New( display(), &std::cout );
}
