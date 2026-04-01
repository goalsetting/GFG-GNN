swig -c\+\+ -python pyabcore.i
python /home/user/GFGGNN/queryopt/setup.py build_ext --inplace
python /home/user/GFDN/queryopt/test.py
cp ./_pyabcore* pyabcore.py ../