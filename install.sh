# current directory
d=`pwd`

# Prompt the user for testing
echo "Do you want to run (a)ll tests, only (f)ast tests?"
read -p "Enter 'a/f' for testing, nothing for skipping" choice

# install
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "No Conda environment is active, installing with --user."
    python3 -m pip install . --user
else
    echo "Conda environment '$CONDA_DEFAULT_ENV' is active."
    python3 -m pip install . 
fi
rm -r ./build

# determine pytest version
if command -v pytest > /dev/null; then
    # pytest is available
    mypytest=pytest
elif command -v pytest-3 > /dev/null; then
    # pytest-3 is available
    mypytest=pytest-3
else
    echo "Neither pytest-3 nor pytest is available. Please install pytest."
fi

# run the appropriate tests
if [[ "$choice" == "a" || "$choice" == "A" ]]; then
    echo "Running all tests..."
    $mypytest --durations=10 --pyargs nanocavity
elif [[ "$choice" == "f" || "$choice" == "F" ]]; then
    echo "Running only fast tests..."
    $mypytest --durations=10 -m "not slow" --pyargs nanocavity
fi
