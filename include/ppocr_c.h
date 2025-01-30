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
typedef struct PPOcrResult * PPPOcrResult;

struct PPOcrResult {
  char* text;
  float score = -1.0;
  float cls_score;
  int cls_label = -1;
  PPPOcrResult next;
};

extern CPPOCR_API int ppocr_from_args( CPPOCR * cppocr, int argc, char ** argv );
extern CPPOCR_API int ppocr_from_inis( CPPOCR * cppocr, char  const * inis );
extern CPPOCR_API int ppocr_from_sxml( CPPOCR * cppocr, char const * xmlfile );

extern CPPOCR_API int ppocr_cmd( CPPOCR cppocr, PPPOcrResult * result = nullptr );
extern CPPOCR_API int ppocr_exe( CPPOCR cppocr, char const * image_dir,
                                 PPPOcrResult * results = nullptr );

extern CPPOCR_API void ppocr_print_result( PPPOcrResult result );

extern CPPOCR_API void ppocr_free( PPPOcrResult result );
extern CPPOCR_API void ppocr_destroy( CPPOCR cppocr );

#ifdef __cplusplus
}
#endif

#endif // _PPOCR_C_API_H_
