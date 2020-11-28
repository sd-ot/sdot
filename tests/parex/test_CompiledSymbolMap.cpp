#include <parex/CompiledSymbolMap.h>
#include <parex/P.h>

// using namespace asimd;

class TestCSM : public CompiledSymbolMap {
protected:
    virtual path output_directory( const std::string &/*parameters*/ ) const override {
        return "objects/test";
    }

    virtual void make_srcs( SrcSet &ff ) const override {
        ff.new_cpp() << "extern \"C\" int " << ff.symbol_name << "() { return " + ff.parameters + "; }";
    }
};

int main() {
    TestCSM test_csm;
    P( test_csm.symbol_for<int(void)>( "18" )() );
}
