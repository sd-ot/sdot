#pragma once

#include <vector>
#include <string>

/**
 argv:
 * filename => name of the file to get `previous_values` at start. At the end (during destruction), save current values in this file
 * --random => get random values
 * --inc    => get next `previous_values`
*/
class OptParm {
public:
    struct             Value       { std::size_t val, max; };

    /**/               OptParm     ( std::string filename = {}, bool random = false );

    double             completion  () const;
    std::size_t        get_value   ( std::size_t max, int loc_random = 0 );
    void               restart     ();
    void               save        ( std::string filename );
    bool               inc         ( bool from_current = true ); ///< return false if finished

    std::vector<Value> previous_values;
    std::vector<Value> current_values;
    std::size_t        random;
    std::size_t        count;
};

