# Info

folder of unit test for the RL-Framework code.
The unit test should be split into groups of no more than 8.
This is a limitation of tensorflow that cuases issues when worker threads are reused for new testcases.

### Dependancies

1. pip3 install --user junit2html
1. pip3 install --user pytest-parallel
1. pip3 install --user pytest-timeout
1. pip3 install --user pytest-xdist

## How to use

```
python3 -m pytest tests/
```

To speed things up and use a lot more memory you can multi process the testing with
```
python3 -m pytest <test_file_path> -n <number_of_processes_to_use> --junitxml= <name_for_test_output_xml>
```
example:
```
python3 -m pytest tests/test_viz_imitation.py -n 4 --junitxml=text_viz.xml
```


### Email test results

```
python3 -m pytest tests/test_saveandload.py -n 3 

convert xml to html and email

https://gitlab.com/inorton/junit2html