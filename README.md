# CycleGAN implementation in Tensorflow 2.2 with custom training loop for CPU/multi GPU/vCPU MirroredStrategy.

Tested on CPU, vCPU and Nvidia DGX GPU(s).

Please check following links:
* CycleGAN: 
  - https://github.com/dragen1860/TensorFlow-2.x-Tutorials/tree/master/15-CycleGAN
* Tensorflow 2.0 multi-GPU:
  - https://www.tensorflow.org/guide/distributed_training#using_tfdistributestrategy_with_custom_training_loops
* Tensorflow virtual GPUs: 
  - https://www.tensorflow.org/guide/gpu
* Datasets:
  - https://people.eecs.berkeley.edu/%7Etaesung_park/CycleGAN/datasets/

## File descriptions

* __init__.py - entry point (***UPDATED - new, more stable version***)
* tfw.py - training framework containing whole logic for training models (***REMOVED***)
* model.py - CycleGAN classes (***REMOVED***)
* data.py - pipelines (***REMOVED***)
* save_load.py - loading and saving models (***REMOVED***)

## Directory descriptions

* images - contains trainA and trainB subdirectories from https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/horse2zebra.zip
* checkpoints - contains saved models
