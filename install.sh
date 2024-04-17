# current directory
d=`pwd`

# install
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "No Conda environment is active, installing with --user."
    python3 -m pip install . --user
else
    echo "Conda environment '$CONDA_DEFAULT_ENV' is active."
    python3 -m pip install .
fi
rm -r ./build

# run tests
cd ~
if command -v pytest > /dev/null; then
    # pytest is available
    pytest --pyargs nanocavity
elif command -v pytest-3 > /dev/null; then
    # pytest-3 is available
    pytest-3 --pyargs nanocavity
else
    echo "Neither pytest-3 nor pytest is available. Please install pytest."
fi
cd $d
