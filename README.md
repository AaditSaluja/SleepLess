# SleepLess

Playground is the emulator. To run, just going into the folder and running

```
python -m experiments.run configs/overlap.yml
```

works. Here, overlap.yml is the config file that you can play around with. The simulator in under env, and for now I have just been manually piping the results from simulator to the playground. Further, for diminishing returns SleepLess implementation, it's under SleepLess2.py and has it's own separate base. Needed to rebuild too much of the playground for it so that exists as a separate thing for now but it does work.

Finally, a host of baseline tests are in the zip with a lot of empirical data files. They are also summarized under the csv in /env!
