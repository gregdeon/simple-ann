# simple-ann
A barebones, simple-as-possible neural network module in Python.

## This Project ##
There are 3 code files in this project:
* `src/net.py`, which contains the neural network code
* `test/train_net.py`, which is a simple script to test the neural network
* `main.py`, which runs the above test

Sample output:
```
> python main.py
Error: 51.8225887749
Error: 38.7242610863
Error: 29.506248854
Error: 26.2518369158
Error: 22.1026533993
Error: 24.6477324424
Error: 25.1741366081
Error: 23.7850999672
Error: 20.3820896864
...
<A lot more lines here...>
...
Error: 0.0412107759986
Error: 0.0418174449554
Error: 0.0401139655417
Error: 0.0405060366548
Error: 0.042101412581
```

## Potential Improvements ##
There are a lot of libraries that have far more features implemented, and this module does not attempt to emulate them. However, some things that would be nice:
* Variable number of hidden layers
* Different training algorithms
* More error-catching
