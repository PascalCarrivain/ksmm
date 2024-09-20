#!/bin/bash

./long_run.sh "kernel_bs_first_float4" 0 "sanity_check" 10 0;
./long_run.sh "kernel_bs_last_float4" 0 "sanity_check" 10 0;

./long_run.sh "kernel_bs_first_half2" 0 "sanity_check" 10 0;
./long_run.sh "kernel_bs_last_half2" 0 "sanity_check" 10 0;
