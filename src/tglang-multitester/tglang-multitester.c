#include "tglang.h"

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h> 

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "Usage: tglang-multitester <input_directory>\n");
    return 1;
  }

  char * dir_name = argv[1];

  // truncate trailing slash
  size_t dir_len = strlen(dir_name);
  for (;dir_len > 0 && dir_name[dir_len-1] == '/';dir_len--) {
    dir_name[dir_len-1] = '\0';
  }

  if (dir_len == 0) {
    fprintf(stderr, "Empty directory name '%s'\n", dir_name);
    return 3;
  }

  DIR *d;
  d = opendir(dir_name);
  if (!d) {
    fprintf(stderr, "Can't open directory '%s'\n", dir_name);
    return 4;
  }
  struct dirent *dir;

  while ((dir = readdir(d)) != NULL)  {
    if (dir->d_type != DT_REG) {
      continue;
    }

    size_t const fname_max_len = 1024;
    char fname[fname_max_len];
    char const * fmt = "%s/%s\0";
    size_t fmt_len = strlen(fmt);
    if (dir_len + fmt_len + strlen(dir->d_name) > fname_max_len) {
      fprintf(stderr, "Too long name, skipping file '%s'\n", dir->d_name);
      continue;
    }
    sprintf(fname, fmt, dir_name, dir->d_name);

    FILE *in = fopen(fname, "rb");
    if (in == NULL) {
      fprintf(stderr, "Failed to open input file %s\n", argv[1]);
      return 1;
    }

    fseek(in, 0, SEEK_END);
    long fsize = ftell(in);
    fseek(in, 0, SEEK_SET);

    char *text = malloc(fsize + 1);
    fread(text, fsize, 1, in);
    if (ferror(in)) {
      fprintf(stderr, "Failed to read input file %s\n", argv[1]);
      return 1;
    }
    fclose(in);

    text[fsize] = 0;

    clock_t begin = clock();
    int enum_code = tglang_detect_programming_language(text);

    // profile in microseconds
    const double US_IN_SEC = 1000000.0;
    int elapsed_us = (clock()-begin)/(double)(CLOCKS_PER_SEC/US_IN_SEC);
    printf("File: '%s', tag: '%d', file size: %ld, timing us: '%d'\n", fname, enum_code, fsize, elapsed_us);
  }
  
  closedir(d);
  return 0;
}
