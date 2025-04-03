import time
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import functools
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map
import jax.sharding as jshard
import optax  # https://github.com/deepmind/optax
import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox


class Func(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.mlp(y)


class PIDNeuralODE(eqx.Module):
    func: Func

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Func(data_size, width_size, depth, key=key)

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=16**3,
        )
        return solution.ys


class StepToNeuralODE(eqx.Module):
    func: Func

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Func(data_size, width_size, depth, key=key)

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=None,
            y0=y0,
            stepsize_controller=diffrax.StepTo(ts),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=16**3,
        )
        return solution.ys


def _get_data(ts, *, key):
    y0 = jr.uniform(key, (2,), minval=-0.6, maxval=1)

    def f(t, y, args):
        x = y / (1 + y)
        return jnp.stack([x[1], -x[0]], axis=-1)

    solver = diffrax.Tsit5()
    dt0 = 0.1
    saveat = diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(f), solver, ts[0], ts[-1], dt0, y0, saveat=saveat
    )
    ys = sol.ys
    return ys


def get_data(dataset_size, *, key):
    ts = jnp.linspace(0, 10, 100)
    key = jr.split(key, dataset_size)
    ys = jax.vmap(lambda key: _get_data(ts, key=key))(key)
    return ts, ys


def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    while True:
        perm = jr.permutation(key, indices)
        (key,) = jr.split(key, 1)
        start = 0
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size


def run_pmap(
    model,
    dataset_size=2048,
    batch_size=32,
    lr_strategy=(3e-3, 3e-3, 3e-3),
    steps_strategy=(1000, 1000, 1000),
    length_strategy=(0.1, 0.5, 1.0),
    width_size=64,
    depth=2,
    seed=5678,
    plot=True,
    print_every=100,
):
    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key = jr.split(key, 3)

    ts, ys = get_data(dataset_size, key=data_key)
    _, length_size, data_size = ys.shape

    model = model(data_size, width_size, depth, key=model_key)

    # Training loop like normal.
    #
    # Only thing to notice is that up until step 500 we train on only the first 10% of
    # each time series. This is a standard trick to avoid getting caught in a local
    # minimum.

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi):
        y_pred = eqx.filter_vmap(model, in_axes=(0, 0))(ti, yi[:, 0])
        return jnp.mean((yi - y_pred) ** 2)

    def make_step(ti, yi, model, opt_state):
        loss, grads = eqx.filter_pmap(
            grad_loss, in_axes=(None, 0, 0), axis_name="device"
        )(model, ti, yi)
        loss = loss.mean(axis=0)
        grads = jax.tree_util.tree_map(lambda x: x.sum(axis=0), grads)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    counter = 0
    num_devices = len(jax.local_devices())

    for lr, steps, length in zip(lr_strategy, steps_strategy, length_strategy):
        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        # pmap_make_step = eqx.filter_pmap(make_step, in_axes=(0, 0, None, None), axis_name="devices")

        _ts = ts[: int(length_size * length)]
        _ys = ys[:, : int(length_size * length)]
        ti = jnp.array(
            batch_size
            * [
                _ts,
            ]
        )
        ti = ti.reshape(num_devices, batch_size // num_devices, *ti.shape[1:])
        epoch_times = []
        for step, (yi,) in zip(
            range(steps), dataloader((_ys,), batch_size, key=loader_key)
        ):
            yi = yi.reshape(num_devices, batch_size // num_devices, *yi.shape[1:])
            # In the very last epoch, create a trace:
            if counter == 2:
                if step == 50:
                    jax.profiler.start_trace(
                        "./tmp/tensorboard/pmap", create_perfetto_trace=True
                    )
                elif step == 60:
                    jax.profiler.stop_trace()
            else:
                counter += 1
            start = time.time()
            loss, model, opt_state = make_step(ti, yi, model, opt_state)
            end = time.time()
            if (step % print_every) == 0 or step == steps - 1:
                print(f"Step: {step}, Loss: {loss}, Computation time: {end - start}")
            if step == 0:
                compile_time = end - start
            else:
                epoch_times.append(end - start)
    epoch_time = np.mean(epoch_times)
    return ts, ys, model, epoch_time, compile_time, float(loss)


def run_sharded(
    model,
    dataset_size=2048,
    batch_size=512,
    lr_strategy=(3e-3, 3e-3, 3e-3),
    steps_strategy=(1000, 1000, 1000),
    length_strategy=(0.1, 0.5, 1.0),
    width_size=128,
    depth=4,
    seed=5678,
    plot=True,
    print_every=100,
):
    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key = jr.split(key, 3)

    ts, ys = get_data(dataset_size, key=data_key)
    _, length_size, data_size = ys.shape

    model = model(data_size, width_size, depth, key=model_key)

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi):
        y_pred = jax.vmap(model, in_axes=(0, 0))(ti, yi[:, 0])
        return jnp.mean((yi[:, 1:] - y_pred[:, :-1]) ** 2)

    @eqx.filter_jit(donate="all")
    def make_step(ti, yi, model, opt_state, sharding, replicated):
        # Share and replicate these
        model, opt_state = eqx.filter_shard((model, opt_state), replicated)
        # Split these over GPUs
        ti, yi = eqx.filter_shard((ti, yi), sharding)
        loss, grads = grad_loss(model, ti, yi)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        # Sync between GPUs
        model, opt_state = eqx.filter_shard((model, opt_state), replicated)
        return loss, model, opt_state

    num_devices = len(jax.local_devices())
    devices = mesh_utils.create_device_mesh((num_devices,))

    # Data will be split along the batch axis
    mesh = Mesh(devices, axis_names=("batch",))  # naming axes of the mesh
    sharding = NamedSharding(
        mesh,
        P(
            "batch",
        ),
    )  # naming axes of the sharded partition
    replicated = NamedSharding(mesh, P())

    counter = 0
    for lr, steps, length in zip(lr_strategy, steps_strategy, length_strategy):
        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

        _ts = ts[: int(length_size * length)]
        _ys = ys[:, : int(length_size * length)]
        ti = jnp.array(
            batch_size
            * [
                _ts,
            ]
        )
        epoch_times = []
        for step, (yi,) in zip(
            range(steps), dataloader((_ys,), batch_size, key=loader_key)
        ):
            # Only create a trace of training on the full timeseries:
            if counter == 3:
                if step == 50:
                    jax.profiler.start_trace(
                        "./tmp/tensorboard/sharded", create_perfetto_trace=True
                    )
                elif step == 60:
                    jax.profiler.stop_trace()
            else:
                if step == 0:
                    counter += 1
            start = time.time()
            loss, model, opt_state = make_step(
                ti, yi, model, opt_state, sharding, replicated
            )
            end = time.time()
            if (step % print_every) == 0 or step == steps - 1:
                print(f"Step: {step}, Loss: {loss}, Computation time: {end - start}")
            if step == 0:
                compile_time = end - start
            else:
                epoch_times.append(end - start)
    epoch_time = np.mean(epoch_times)
    return ts, ys, model, epoch_time, compile_time, float(loss)


def run_shard_mapped(
    model,
    dataset_size=2048,
    batch_size=512,
    lr_strategy=(3e-3, 3e-3, 3e-3),
    steps_strategy=(1000, 1000, 1000),
    length_strategy=(0.1, 0.5, 1.0),
    width_size=128,
    depth=4,
    seed=5678,
    plot=True,
    print_every=100,
):
    key = jr.PRNGKey(seed)
    data_key, model_key, loader_key = jr.split(key, 3)

    ts, ys = get_data(dataset_size, key=data_key)
    _, length_size, data_size = ys.shape

    model = model(data_size, width_size, depth, key=model_key)

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi):
        y_pred = shard_map(
            jax.vmap(model, in_axes=(0, 0)),
            mesh=mesh,
            in_specs=spec,
            out_specs=spec,
            check_rep=False,
        )(ti, yi[:, 0])
        return jnp.mean((yi[:, 1:] - y_pred[:, :-1]) ** 2)

    @eqx.filter_jit(donate="all")
    def make_step(ti, yi, model, opt_state, sharding, replicated):
        # Share and replicate these
        model, opt_state = eqx.filter_shard((model, opt_state), replicated)
        # Split these over GPUs
        ti, yi = jax.device_put((ti, yi), sharding)
        loss, grads = grad_loss(model, ti, yi)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        # Sync between GPUs
        # model, opt_state = eqx.filter_shard((model, opt_state), replicated)
        return loss, model, opt_state

    num_devices = len(jax.local_devices())
    devices = mesh_utils.create_device_mesh((num_devices,))

    # Data will be split along the batch axis
    mesh = Mesh(devices, axis_names=("batch",))  # naming axes of the mesh
    spec = P("batch")
    sharding = NamedSharding(mesh, spec)  # naming axes of the sharded partition
    replicated = NamedSharding(mesh, P())

    counter = 0
    for lr, steps, length in zip(lr_strategy, steps_strategy, length_strategy):
        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

        _ts = ts[: int(length_size * length)]
        _ys = ys[:, : int(length_size * length)]
        ti = jnp.array(
            batch_size
            * [
                _ts,
            ]
        )
        epoch_times = []
        for step, (yi,) in zip(
            range(steps), dataloader((_ys,), batch_size, key=loader_key)
        ):
            # Only create a trace of training on the full timeseries:
            if counter == 3:
                if step == 50:
                    jax.profiler.start_trace(
                        "./tmp/tensorboard/shmap", create_perfetto_trace=True
                    )
                elif step == 60:
                    jax.profiler.stop_trace()
            else:
                if step == 0:
                    counter += 1
            start = time.time()
            loss, model, opt_state = make_step(
                ti, yi, model, opt_state, sharding, replicated
            )
            end = time.time()
            if (step % print_every) == 0 or step == steps - 1:
                print(f"Step: {step}, Loss: {loss}, Computation time: {end - start}")
            if step == 0:
                compile_time = end - start
            else:
                epoch_times.append(end - start)
    epoch_time = np.mean(epoch_times)
    return ts, ys, model, epoch_time, compile_time, float(loss)


if __name__ == "__main__":
    params = dict(
        dataset_size=2048,
        batch_size=32,
        lr_strategy=(3e-3, 3e-3, 3e-3),
        steps_strategy=(1000, 1000, 1000),
        length_strategy=(0.1, 0.5, 1.0),
        width_size=64,
        depth=2,
        seed=5678,
        print_every=100,
    )

    parser = argparse.ArgumentParser(
        description="Configuration for diffrax benchmarking"
    )
    parser.add_argument(
        "-d", "--device", type=str, help="Device to run the NeuralODE on", default="cpu"
    )
    parser.add_argument(
        "-n",
        "--number_of_devices",
        type=int,
        help="Number of devices to use",
        default=1,
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        help="Path to the configuration yaml file",
        default="shard",
    )
    parser.add_argument(
        "-c",
        "--controller",
        type=str,
        help="The controller strategy to use",
        default="pid",
    )
    args = vars(parser.parse_args())
    if args["device"] == "gpu":
        # Set the number of GPUS, only works for NVIDIA right now:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            [str(i) for i in range(args["number_of_devices"])]
        )
        # Check that we are actually using GPU
        assert jax.default_backend() == "gpu", (
            f"GPU chosen as backend, but it is not available, instead have {jax.default_backend()}"
        )
        # Check that the number of GPUS is right
        assert args["number_of_devices"] == len(jax.local_devices()), (
            f"Tried to set the number of GPUS, but failed, goal: {args['number_of_devices']}, have {jax.local_devices()}"
        )
    elif args["device"] == "cpu":
        # Set the number of CPUs
        os.environ["XLA_FLAGS"] = (
            f"--xla_force_host_platform_device_count={args['number_of_devices']}"
        )
        # Set the Jax device to CPU:
        jax.config.update("jax_platform_name", "cpu")
        # Check that we use CPU:
        assert jax.default_backend() == "cpu", (
            f"CPU chosen as backend, but it is not available, instead have: {jax.default_backend()}"
        )
        # Check that the number of CPUS is right:
        assert args["number_of_devices"] == len(jax.local_devices()), (
            f"Tried to set the number of CPUs, but failed, goal: {args['number_of_devices']}, have {jax.local_devices()}"
        )
    else:
        RuntimeError("Choose between 'gpu' or 'cpu'")

    print(jax.print_environment_info())
    if args["controller"] == "stepto":
        model = StepToNeuralODE
    elif args["controller"] == "pid":
        model = PIDNeuralODE
    else:
        RuntimeError("Choose between 'stepto' or 'pid'")

    start_time = time.time()
    if args["mode"] == "shard":
        ts, ys, model, epoch_time, compile_time, final_loss = run_sharded(
            model, **params
        )
    elif args["mode"] == "pmap":
        ts, ys, model, epoch_time, compile_time, final_loss = run_pmap(model, **params)
    elif args["mode"] == "shmap":
        ts, ys, model, epoch_time, compile_time, final_loss = run_shard_mapped(
            model, **params
        )
    else:
        RuntimeError("Choose between 'shard' or 'pmap'")
    runtime = time.time() - start_time
    with open(f"runs_{args['device']}.txt", "a") as fh:
        fh.write(
            f"{args}, {runtime=}, {epoch_time=}, {compile_time=}, {final_loss=}" + "\n"
        )
