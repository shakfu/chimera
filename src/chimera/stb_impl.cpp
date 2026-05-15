/*
 * stb_impl.cpp - Implementation file for stb_image_write.
 *
 * The read side (stbi_load / stbi_image_free) is intentionally not defined
 * here: libmtmd.a's mtmd-helper.o ships its own STB_IMAGE_IMPLEMENTATION
 * with non-static symbols, so providing them here too would yield a
 * duplicate-symbol link error once any mtmd_helper_* function is referenced.
 * chimera_sd.cpp's stbi_load calls resolve against libmtmd's copy instead.
 *
 * stb_image_write is NOT provided by libmtmd, so we still own that side.
 */

#include <algorithm>  // stb_image_write uses std::min in C++ mode

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
