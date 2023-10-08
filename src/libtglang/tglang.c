#include "tglang.h"

#include <stdlib.h>
#include <string.h>

enum TglangLanguage tglang_detect_programming_language(const char *text) {
  if (strstr(text, "std::") != NULL) {
    return TGLANG_LANGUAGE_CPLUSPLUS;
  }
  if (strstr(text, "let ") != NULL) {
    return TGLANG_LANGUAGE_JAVASCRIPT;
  }
  if (strstr(text, "int ") != NULL) {
    return TGLANG_LANGUAGE_C;
  }
  if (strstr(text, ";") == NULL) {
    return TGLANG_LANGUAGE_PYTHON;
  }
  return TGLANG_LANGUAGE_OTHER;
}
