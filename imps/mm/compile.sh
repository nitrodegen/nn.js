#!/bin/bash

emcc mm.c -o mm.js -s EXPORTED_FUNCTIONS="['_dot_prod_single','_dbg_arr','_init_mat','_compute_linear_layer','_malloc','_free']" -s EXPORTED_RUNTIME_METHODS=["ccall","cwrap"]  -s BINARYEN_ASYNC_COMPILATION=0

