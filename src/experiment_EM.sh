#!/bin/bash
: '
Just run the EM algorithm a bunch of times with various n-samples
$1 = save data number
$2 = serial
'
dir="$(dirname "${BASH_SOURCE[0]}")"
dir="$(realpath "${dir}")"
serial=$2
if [[ "$serial" != "" ]]; then
  serial="_$serial"
fi

rm ${dir}/em_seed.jld2
for i in 10000 30000 100000 300000
do
  julia --project=${dir}/.. ${dir}/main.jl EM ${dir}/save_data_${1}.jld2 -n $i > ${dir}/plot_${1}${serial}_$i.txt
done
