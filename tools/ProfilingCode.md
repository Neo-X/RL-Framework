

python -m cProfile -o myscript.cprof myscript.py
pyprof2calltree -k -i myscript.cprof