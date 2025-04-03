# Multi GPU performance of NeuralODEs
! On CUDA jaxlib >0.5.0 there seem to be even more performance issues, so use jax 0.4.34 or earlier.


## Super quick NeuralODES
We want to approximate time series data of the nature $X(t)$, with $n$ entries for $T$ timesteps. Instead of trying to learn this directly with a Neural Network (full sequence) or with a recurrent architecture $x_t \rightarrow x_{t+1}$, we solve a differential equation with a neural network on its
right hand side:
$$ \tilde{x}_t = \mathrm{ODESolve}(\mathrm{NN}(x_0, t))$$ 
with initial state $x_0$. This can be trained effectively using the adjoint state.

## Problem statement
In order to train NeuralODEs with large data volumes, we need to have some kind of
data parallel way of training on multiple GPUS. Sadly, there are currently two performance
problems that hamper training as of now:

### No asynchronous DDP
At the moment, there is no asynchronous
way of splitting batches in Jax as far as I am aware of, which means that all GPUS run in
lock-step with eachother. This is suboptimal since the number of steps one needs to take
to accurately solve the latent differential equation can vary depending on both the
sampling, length and complexity of the series we try to reproduce. This results in 
sub-batches being finished quickly on some GPUS, waiting for the most "complex" batch
to finish, only then syncing the gradients and proceeding to the next batch.

### Forced GPU communication
Unfortunately there is another problem that is even more deterimental to performance:
the GPUS try to synchronize the SaveAt state between eachother, 
resulting in many unnescessary additional GPU communication
via AllReduce statements. 

The problem is the following, we solve some differential equation as a function
of some time parameter: `x(t) = ODESOLVE(NN(x_0, t))`, with us providing a sampling train
`ts=[0.0, 0.01, 0.02, ... 1.0]` resulting in 101 samples of time. These timesteps however
have little meaning to the ODE solver, since we want it to choose its own sensible 
timesteps based on a PID controller, resulting in an efficient solution process with
a minimal number of steps. The algorithm then populates our sampling steps with 
values, checking if we either need to proceed to the next solver timestep
or our next sampler timestep.

Now when we choose to instead force our solver to take our sampler steps as timesteps,
the multi-gpu problem vanishes and the (weak) scaling again becomes (sub)`linear`. Unfortunately,
the solver steps that we choose to sample at, are often not ideal for solving the differential equation,
resulting in a loss of performance larger than we could gain with the 

## Testing Jax parallelism 
We can use pmap as a way to do parallel compute as well as sharding. Together with the 
solvers this gives us a 6 scenarios to run:
- single GPU PID
- single GPU StepTo
- multi GPU PID pmap 
- multi GPU PID sharded
- multi GPU StepTo pmap
- multi GPU StepTo sharded

We can then repeat all these experiments on CPU as a benchmark.

The results from `test_all_neuralode.py`:




### GPU
| Device | Number of Devices | Mode  | Controller | Runtime (s) | Epoch Time (s) | Compile Time (s) | Final Loss |
|--------|-------------------|-------|------------|-------------|-----------------|------------------|------------|
| gpu    | 1                 | shard | pid        | 136.69      | 0.05603         | 5.74328          | 0.00123    |
| gpu    | 1                 | shard | stepto     | 217.65      | 0.10833         | 5.11248          | 0.00087    |
| gpu    | 2                 | pmap  | pid        | 197.40      | 0.06361         | 9.95667          | 0.00023    |
| gpu    | 2                 | shard | pid        | 176.93      | 0.07262         | 6.25344          | 0.00095    |
| gpu    | 2                 | pmap  | stepto     | 288.90      | 0.11663         | 9.14150          | 0.00021    |
| gpu    | 2                 | shard | stepto     | 244.53      | 0.11607         | 5.52316          | 0.00088    |




### CPU
| Device | Number of Devices | Mode  | Controller | Runtime (s)  | Epoch Time (s) | Compile Time (s) | Final Loss |
|--------|-------------------|-------|------------|--------------|----------------|------------------|------------|
| cpu    | 1                 | shard | pid        | 177.26      | 0.07678         | 6.00658          | 0.00094    |
| cpu    | 1                 | shard | stepto     | 430.40      | 0.24678         | 5.12502          | 0.00089    |
| cpu    | 8                 | pmap  | pid        | 130.67      | 0.02053         | 10.15768         | 0.00025    |
| cpu    | 8                 | shard | pid        | 1187.57     | 0.69745         | 6.76410          | 0.00093    |
| cpu    | 8                 | pmap  | stepto     | 203.89      | 0.05965         | 9.00977          | 0.00025    |
| cpu    | 8                 | shard | stepto     | 589.55      | 0.32672         | 5.46397          | 0.00089    |


The underlying issue is that Diffrax internally uses a double while loop (to also support SDEs) and XLA doesn't play nice with this. Additionally the SaveAt system introduces a lot of collective calls:
References:
Issue on Diffrax: 
- discussion on slow sharding on Diffrax: https://github.com/patrick-kidger/diffrax/issues/407
- MVP on slow sharding on jax: https://github.com/jax-ml/jax/issues/20968
- MVP sharding vs pmap on JAX: https://github.com/jax-ml/jax/issues/26586
- Additional slowdown of pmap as of 0.4.33: https://github.com/openxla/xla/issues/23110 