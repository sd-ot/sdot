#pragma once

#define SDOT_CONCAT_TOKEN_2( foo, bar ) SDOT_CONCAT_TOKEN_IMPL_2( foo, bar )
#define SDOT_CONCAT_TOKEN_IMPL_2( foo, bar ) foo##bar

#define SDOT_CONCAT_TOKEN_3( foo, bar, baz ) SDOT_CONCAT_TOKEN_IMPL_3( foo, bar, baz )
#define SDOT_CONCAT_TOKEN_IMPL_3( foo, bar, baz ) foo##bar##baz

#define SDOT_CONCAT_TOKEN_4( foo, bar, baz, baa ) SDOT_CONCAT_TOKEN_IMPL_4( foo, bar, baz, baa )
#define SDOT_CONCAT_TOKEN_IMPL_4( foo, bar, baz, baa ) foo##bar##baz##baa

#define SDOT_CONCAT_TOKEN_4_( foo, bar, baz, baa ) SDOT_CONCAT_TOKEN_IMPL_4_( foo, bar, baz, baa )
#define SDOT_CONCAT_TOKEN_IMPL_4_( foo, bar, baz, baa ) foo##_##bar##_##baz##_##baa

#define SDOT_STRINGIFY( foo ) SDOT_STRINGIFY_IMPL( foo )
#define SDOT_STRINGIFY_IMPL( foo ) #foo

