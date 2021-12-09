# random_walk

random walk kernel of the https://github.com/talnish/iiswc21_rwalk repository.

1) Clone the repository:
`git clone https://github.com/arkhadem/random_walk.git`

2) CD to the repo directory:
`cd random_walk`

3) Build the executable code:
`./build`

4) Run the random walk kernel:
`sbatch run_test tgraph.wel`

You can make larger synthetic graphs using this script:

`python generate_synthetic.py -n #nodes# -e #edges# -s #seed#`

## build
#### great lakes
* run ```module load cuda``` and ```module load cmake``` to load environment
* run ```./build.sh rwalk```, you shall find an excutable at root directory called ```random_walk```
* to generate output, run ```./run_tests *.wel > out.txt``` where ```*.wel``` is your graph file. You shall have ```out.txt``` demonstrate the execution time and ```out_random_walk.txt``` as the generated random walk output.

####
We've built the file into binary, you can also directly run ```./run_tests synth_N_10_E_50_S_42_preproc.wel > out.txt``` with our toy example.
## outputs
the folder ```out``` contains the test we have run so far.