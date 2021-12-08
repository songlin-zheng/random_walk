#!/bin/bash
set -Eeuo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: ./build prog_name"
    exit 1
fi

rm -rf _build
mkdir _build

if [ "$1" = "rwalk" ]
then
    cp "./CMakeLists_rwalk.txt" "./CMakeLists.txt"
elif [ "$1" = "helper" ]
then
    cp "./CMakeLists_helper.txt" "./CMakeLists.txt"
elif [ "$1" == "rwalk_op" ]
then
    cp "./CMakeLists_rwalk_optimized.txt" "./CMakeLists.txt"
fi

cd _build/
cmake ..
make -j4
if [ "$1" = "rwalk" ]
then
    mv "./random_walk" "../"
elif [ "$1" == "helper" ]
then
    mv "./helper" "../"
elif [ "$1" == "rwalk_op" ]
then
    mv "./rwalk_op" "../"
fi
cd ../

echo "Successfully built"
