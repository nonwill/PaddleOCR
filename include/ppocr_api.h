#ifndef PPOCR_API_HH
#define PPOCR_API_HH

#ifdef WIN32
#ifdef PPOCR_LIBRARY
#define PPOCR_API __declspec(dllexport)
#else
#define PPOCR_API __declspec(dllimport)
#endif
#else
#define PPOCR_API
#endif

#endif // PPOCR_API_HH
