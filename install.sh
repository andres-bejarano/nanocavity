python setup.py sdist

cd \dist

pip install nanocavity-0.1.tar.gz

cd ..\

pytest nanocavity

