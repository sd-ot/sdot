#pragma once

#include <vector>
#include <deque>
#include <array>
#include <map>

/***/
class GenCutCaseWriter {
public:
    using                     TI              = std::size_t;
    using                     Node            = std::array<TI,2>;
    using                     SrcMap          = std::map<TI,TI>;

    /**/                      GenCutCaseWriter( const GenCutCaseWriter &that );
    /**/                      GenCutCaseWriter();

    void                      operator=       ( const GenCutCaseWriter &that );

    void                      add_output      ( std::string shape_name, std::vector<Node> nodes );
    void                      optimize        ();

    std::string               func_name       () const;
    void                      write           ( std::ostream &os, TI num_case );

private:
    struct                    Output          {
        std::string           func_name       () const;
        Node                  orig_node       ( TI i ) const;

        std::vector<TI>*      num_dst_vertex;
        SrcMap*               src_map;
        std::vector<Node>     nodes;
    };

    struct                    ByOutputShape   {
        std::string           func_name       ( TI num_output_shape = 0 ) const;
        std::string           func_args       () const;
        void                  optimize        ();

        std::vector<TI>       num_dst_vertex;
        std::string           shape_name;
        SrcMap*               src_map;
        std::vector<Output>   outputs;
    };

    ByOutputShape&            by_output_shape ( std::string shape_name, TI nb_nodes );
    std::string               func_args       ( TI num_case ) const;
    //    void                      write_func_name ( std::ostream &os, const ByOutputShape &bos, TI n ) const;
    //    void                      write_func_name ( std::ostream &os ) const;
    //    void                      write_func_args ( std::ostream &os, TI num_case ) const;
    //    bool                      operator<       ( const GenCutCaseWriter &that ) const;

    //    void                      optimize_src_map();
    //    void                      optimize        ( ByOutputShape &bos );

    std::deque<ByOutputShape> by_output_shapes;
    std::map<TI,TI>           src_map;
};

