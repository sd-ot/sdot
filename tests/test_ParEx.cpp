#include "../src/sdot/ParEx/Kernel.h"
#include "../src/sdot/support/P.h"

using namespace parex;
using TF = double;
using TI = int;

int main() {
    // Uniform[ "FP64" ]
    // MinMax[ dim ]
    // Display
    // MinMax[ dim ]
    int dim = 2;
    int nb_diracs = 200;
    Node p = Node::New( uniform( "FP64" ), dim * nb_diracs );
}
