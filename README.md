## Code Structure

+ algorithm
  
  + zoo: zeroth order optimization algorithms
    
    + gradient_estimate.py: single process ZO-CGE, ZO-Quantized RGE and CGE
    
+ data: getting data loaders and the class number information, usage: `from data import prepare_dataset`

+ experiments: main executable files

  + train.py: traning the CNN on cifar10 using the selected method. example usage: `python3 experiments/train.py --lr 0.01 --epoch 50 --batch-size 256 --cnn-depth 2 --cnn-channel-num 18 --method ZO_CGE_Q --num-bits 8`

  + Available training methods: `FO`, `ZO_CGE`, `ZO_CGE_Q`, `ZO_RGE_Q`
  + `FO` - First Order using SGD (relevant cli arguments - weight-decay, momentum, nesterov, scheduler) 
  + `ZO_CGE` - Zero-Order CGE (relevant cli arguments - zoo-step-size) 
  + `ZO_CGE_Q/ZO_RGE_Q` - Zero-Order CGE/RGE with Quantization (relevant cli arguments - num_bits)
  
+ models:

  + cnn.py - Simple CNN with configurable depth (relevant cli arguments cnn_depth argument)
  
+ tools

  + cfg.py: path configs
  + use `--log` to save training data into `results_path/results`
