#!/bin/bash
: '
This script just runs main.jl repeatedly in terse mode with various checkpoints and print the result
'
dir="$(dirname "${BASH_SOURCE[0]}")"
dir="$(realpath "${dir}")"
for i in {7..21}
do
  julia --project=${dir}/.. ${dir}/main.jl p ${dir}/save_data.jld2 -i $i -t
done
