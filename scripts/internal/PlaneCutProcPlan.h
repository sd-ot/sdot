#pragma once

#include <vector>
#include <bitset>
class GenPlaneCutProc;
class OptParm;

/**
*/
class PlaneCutProcPlan {
public:
    struct            Item            { int n0, n1 = -1; };

    /**/              PlaneCutProcPlan( int old_nb_nodes, std::bitset<32> outside_nodes );

    void              write_code      ( std::ostream &os, const std::string &sp, GenPlaneCutProc &gp, OptParm &op );
    std::vector<int>  cut_indices     ();
    void              make_svec       ( std::ostream &os, const std::string &sp, std::string name, std::string n0, std::string n1 );
    int               nb_cuts         ();
    std::string       sval              ( GenPlaneCutProc &gp, std::string n, int ind );

    std::bitset<32>   outside_nodes;
    int               old_nb_nodes;
    std::vector<Item> new_nodes;
};

