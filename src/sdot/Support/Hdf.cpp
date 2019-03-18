#include "Hdf.h"

//// nsmake lib_path /usr/lib/x86_64-linux-gnu/hdf5/serial
//// nsmake lib_name hdf5

using namespace std;

Hdf::Hdf() : h5_file( 0 ) {
}

Hdf::Hdf( string filename, bool clear_old, bool read_only ) : h5_file( 0 ) {
    open( filename, clear_old );
}

Hdf::~Hdf() {
    if ( h5_file )
        H5Fclose( h5_file );
}

void Hdf::write_to_stream(ostream &os) const {
    os << h5_file;
}

void Hdf::open( std::string filename, bool clear_old, bool read_only ) {
    if ( h5_file )
        H5Fclose( h5_file );
    // create or read a previously created file
    if ( clear_old )
        h5_file = H5Fcreate( filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT );
    else
        h5_file = H5Fopen( filename.c_str(), read_only ? H5F_ACC_RDONLY : ( H5F_ACC_RDWR | H5F_ACC_CREAT ), H5P_DEFAULT );
}

static herr_t file_info( hid_t, const char *name, void *op_data ) {
    reinterpret_cast<std::vector<std::string> *>( op_data )->push_back( name );
    return 0;
}

std::vector<std::string> Hdf::list_dir( const std::string &dir ) const {
    std::vector<std::string> res;
    H5Giterate( h5_file, dir.c_str(), NULL, file_info, &res );
    return res;
}

void Hdf::add_tag(const string &name, const string &tag, const char *tag_value) {
    hid_t dataset = H5Gopen1( h5_file, name.c_str() );
    hid_t aid     = H5Screate( H5S_SCALAR );
    hid_t atype   = H5Tcopy( H5T_C_S1 );

    // variables size
    H5Tset_size( atype, H5T_VARIABLE );
    hid_t attr = H5Acreate1( dataset, tag.c_str(), atype, aid, H5P_DEFAULT );
    H5Awrite( attr, atype, &tag_value );

    H5Sclose( aid     );
    H5Aclose( attr    );
    H5Gclose( dataset );
}

void Hdf::add_tag(const string &name, const string &tag, const string &tag_value) {
    add_tag( name, tag, tag_value.c_str() );
}

void Hdf::read_tag( const std::string &name, const std::string &tag, std::string &tag_value, bool group ) const {
    hid_t dataset = group ? H5Gopen1( h5_file, name.c_str() ) : H5Dopen1( h5_file, name.c_str() );
    hid_t attr    = H5Aopen_name( dataset, tag.c_str() );
    hid_t ftype   = H5Aget_type( attr );
    hid_t atype   = H5Tget_native_type( ftype, H5T_DIR_ASCEND );

    if ( H5Tget_strpad( ftype ) == H5T_STR_NULLTERM ) {
        char *string_attr;
        H5Aread( attr, atype, &string_attr );
        if ( string_attr )
            tag_value = string_attr;
        else
            tag_value = std::string();
        ::free( string_attr );
    } else {
        size_t size   = H5Tget_size( ftype );

        char *string_attr = (char *)std::malloc( size );
        H5Aread( attr, atype, string_attr );
        tag_value = string_attr;
        std::free( string_attr );
    }

    H5Aclose( attr    );
    H5Tclose( atype   );
    if ( group )
        H5Gclose( dataset );
    else
        H5Dclose( dataset );
}

void Hdf::read_group_size( const std::string &name, int &size ) const {
    hid_t dataset = H5Gopen1( h5_file, name.c_str() );

    H5G_info_t group_info;
    H5Gget_info( dataset, &group_info );
    size = group_info.nlinks;

    H5Gclose(dataset);
}

void Hdf::read_size( const std::string &name, size_t &size ) const {
    std::vector<size_t> size_vec;
    read_size( name, size_vec );
    size = size_vec[ 0 ];
}

bool Hdf::dataset_exist( const string &name ) {
    bool exist = false;
    if ( H5Lexists( h5_file , name.c_str(), H5P_DEFAULT ) ){
        exist = true;
    }
    return exist;
}
