#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: ./make_pypy_release.sh X.Y.Z"
    exit 2
fi

source ./make_documentation.sh --commit
git push

# Make release
cd ..
git tag $1 -m "heyvi-$1"
git push --tags origin main

python3 setup.py sdist bdist_wheel
twine upload dist/*

rm -rf dist/
rm -rf build/
rm -rf heyvi.egg-info/
