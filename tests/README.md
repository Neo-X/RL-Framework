## How to use

```
python3 -m pytest tests/
```

To speed things up and use a lot more memory you can multi process the testing with
```
python3 -m pytest tests/ -n<number_of_processes_to_use>
```

### Dependancies

1. pip3 install --user pytest-xdist
1. pip3 install --user pytest