## How to use

```
python3 -m pytest tests/
```

To speed things up and use a lot more memory you can multi process the testing with
```
python3 -m pytest tests/ -n<number_of_processes_to_use>
```

### Dependancies

1. pip3 install --user nose
-- nose does not support multiprocessing and xunit need another library
1. pip3 install --user nose_xunitmp
1. pip3 install --user junit2html

for pytest

1. pip3 install --user pytest-parallel
1. pip3 install --user pytest-timeout

### Email test results

```
python3 -m nose tests/ 

convert xml to html and email

https://gitlab.com/inorton/junit2html