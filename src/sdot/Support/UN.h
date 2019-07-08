#ifdef UN
#elif defined(__GNUC__)
# define UN(x) (void)x // __attribute__((unused))
#elif defined(__LCLINT__)
# define UN(x) /*@unused@*/ x
#elif defined(__cplusplus)
# define UN(x)
#else
# define UN(x) x
#endif
