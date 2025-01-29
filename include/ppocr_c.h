#ifndef _PPOCR_C_API_H_
#define _PPOCR_C_API_H_

#ifdef WIN32
#   ifdef CPPOCR_LIBRARY
#       define CPPOCR_API __declspec(dllexport)
#   else
#       define CPPOCR_API __declspec(dllimport)
#   endif
#else
#   define CPPOCR_API
# endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct PPOCRC * CPPOCR;

extern CPPOCR_API int ppocr_from_args( CPPOCR * cppocr, int argc, char ** argv );
extern CPPOCR_API int ppocr_from_sxml( CPPOCR * cppocr, char const * xmlfile );
extern CPPOCR_API void ppocr_enable_cout( CPPOCR cppocr, bool yes=true );
extern CPPOCR_API void ppocr_destroy( CPPOCR cppocr );

extern CPPOCR_API int ppocr_exe( CPPOCR cppocr, char const * image_dir );
extern CPPOCR_API int ppocr_cmd( CPPOCR cppocr );

#ifdef __cplusplus
}
#endif

#endif // _PPOCR_C_API_H_
