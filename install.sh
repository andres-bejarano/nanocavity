# current directory
d=`pwd`

# install
python3 -m pip install . --user

# run tests
cd ~
pytest-3 --pyargs nanocavity
cd $d
