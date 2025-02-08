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
  float score;
  float cls_score;
  int cls_label;
  PPPOcrResult next;
};

extern CPPOCR_API int ppocr_from_args( CPPOCR * cppocr, int argc, char ** argv ) noexcept;
extern CPPOCR_API int ppocr_from_inis( CPPOCR * cppocr, char const * inis ) noexcept;
extern CPPOCR_API int ppocr_from_sxml( CPPOCR * cppocr, char const * xmlfile ) noexcept;

extern CPPOCR_API int ppocr_cmd( CPPOCR cppocr, PPPOcrResult * result = nullptr ) noexcept;
extern CPPOCR_API int ppocr_exe( CPPOCR cppocr, char const * image_dir,
                                 PPPOcrResult * results = nullptr ) noexcept;

extern CPPOCR_API void ppocr_print_result( PPPOcrResult result ) noexcept;

extern CPPOCR_API void ppocr_free( PPPOcrResult result ) noexcept;
extern CPPOCR_API void ppocr_destroy( CPPOCR cppocr ) noexcept;

extern CPPOCR_API void ppocr_print_help() noexcept;

#ifdef __cplusplus
}
#endif

#endif // _PPOCR_C_API_H_
