# iclr-reproducability-challenge
Reproducing the results from an ICLR paper on few shot learning. 


## Run Guide

Reproduce results 5 shot learning results.
```
./scripts/5_shot_MIN.sh
```

MNIST Test
Note: save state in the first script must be the same as the loadstate in the second script.
```
./scripts/10_class_1_channel_5_shot_MIN.sh
./scripts/5_shot_MNIST_test.sh 
```

## Authors

* **Aran Smith** [abs2n18@soton.ac.uk]()
* **Juan Olloniego** [jao1n18@soton.ac.uk]()
* **Sulaiman Sadiq** [ss2n18@soton.ac.uk]()
