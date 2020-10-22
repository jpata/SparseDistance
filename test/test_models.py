from sparsedistance.models import SparseHashedNNDistance
from sparsedistance.utils import sparse_dense_matmult_batch

import numpy as np
import tensorflow as tf
import time

def test_eager():
    num_batches = 10
    num_points_per_batch = 1000
    num_features = 32
    
    X = np.array(np.random.randn(num_batches, num_points_per_batch, num_features), dtype=np.float32)
    y = np.array(np.random.randn(num_batches, num_points_per_batch, ), dtype=np.float32)
    
    #show that we can take a gradient of stuff with respect to the distance matrix values (but not indices!)
    dense_transform = tf.keras.layers.Dense(128)
    dm_layer = SparseHashedNNDistance()
    
    with tf.GradientTape(persistent=True) as g:
        X_transformed = dense_transform(X)
        dm = dm_layer(X_transformed)
    
        ret = sparse_dense_matmult_batch(dm, X)
    
        #reduce the output to a single scalar, just for demonstration purposes
        ret = tf.reduce_sum(ret)
    
    grad = g.gradient(ret, dense_transform.weights)
    
    assert(not (grad is None))
    assert(grad[0].numpy().sum()!=0)
    assert(grad[1].numpy().sum()!=0)
    print("eager test successful, sum(dm*X)={}, sum(grad)={}".format(ret, grad[0].numpy().sum()))

def test_timing(bin_size=100, max_num_bins=1000):

    def train_model(num_samples_per_graph, num_features=32, minibatch_size_in_training=1):
        assert(bin_size*max_num_bins >= num_samples_per_graph)
        X = np.random.randn(100, num_samples_per_graph, num_features).astype(np.float32)
        y = np.random.randn(100, num_samples_per_graph, 1).astype(np.float32)

        inputs = tf.keras.Input(shape=(num_samples_per_graph, num_features,), batch_size=minibatch_size_in_training)
        dense = tf.keras.layers.Dense(128, activation=tf.keras.activations.selu)(inputs)
        dense2 = tf.keras.layers.Dense(128, activation=tf.keras.activations.selu)(inputs)

        #compute the distance matrix between points
        #[num_batches, num_samples_per_graph, 128] -> [num_batch, num_samples_per_graph, num_samples_per_graph]
        dm = SparseHashedNNDistance(bin_size=bin_size, max_num_bins=max_num_bins, num_neighbors=10)(dense)

        #compute the dm*x sparsely across the batches:
        #[num_batches, num_samples_per_graph, num_samples_per_graph] x [num_batches, num_samples_per_graph, 128] -> [num_batch, num_samples_per_graph, 128]
        dm_x_dense = tf.keras.layers.Lambda(lambda args: sparse_dense_matmult_batch(args[0], args[1]))([dm, dense2])

        #just a dummy output for each point [num_batches, num_samples_per_graph, num_features]
        dense_out = tf.keras.layers.Dense(1)(dm_x_dense)

        model = tf.keras.Model(inputs=inputs, outputs=[dense_out, dm], name="test")
        model.compile(loss=["mse", None], optimizer="adam")

        t0 = time.time()
        print(X.shape, y.shape)
        ret = model.fit(X, y, batch_size=minibatch_size_in_training, epochs=1, verbose=False)
        t1 = time.time()
        return (t1 - t0)

    num_samples = []
    times = []
    for num_samples_per_graph, minibatch_size in [(1000, 50), (10000, 20), (20000, 10), (50000, 2), (100000, 1)]:
        dt = train_model(num_samples_per_graph, minibatch_size_in_training=minibatch_size)
        num_samples.append(num_samples_per_graph)
        times.append(dt)
        print(num_samples_per_graph, dt)

    num_samples = np.array(num_samples)
    times = np.array(times)
    times /= times[0]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6,5))
    plt.plot(num_samples, times, marker="o")
    plt.xlabel("number of elements per graph [N]")
    plt.ylabel("scaling of the training time:\n[t(N) / t(1000)]")
    plt.title("Scaling of the training time with input size")
    plt.savefig("images/timing.png", dpi=500)

def generate_event(mean_num_particles_per_event=1000, max_particle_energy=10.0, deposit_fraction=0.1, lowest_energy_threshold=0.5, deposit_pos_spread=0.02):
    particles = []
    all_deposits = []
    for ipart in range(np.random.poisson(mean_num_particles_per_event)):
        energy = np.random.uniform(0, max_particle_energy)
        pos_x = np.random.uniform(-1.0, 1.0)
        pos_y = np.random.uniform(-1.0, 1.0)
        orig_energy = energy
        particles.append([orig_energy, pos_x, pos_y])
        deposits = []
        while energy > lowest_energy_threshold:
            deposit_energy = np.random.normal(energy * deposit_fraction)
            if deposit_energy > lowest_energy_threshold:
                energy -= deposit_energy
                deposit_x = np.random.uniform(pos_x-deposit_pos_spread, pos_x+deposit_pos_spread)
                deposit_y = np.random.uniform(pos_y-deposit_pos_spread, pos_y+deposit_pos_spread)
                deposits.append([deposit_energy, deposit_x, deposit_y, -1, ipart])
        if len(deposits) > 0:
            top_deposit_index = np.argsort(np.array([d[0] for d in deposits]))[-1]
            deposits[top_deposit_index][3] = ipart
            all_deposits.append(deposits)


    particles_array = np.stack(particles)
    deposits_array = np.concatenate(all_deposits)
    particles_array_resized = np.zeros((deposits_array.shape[0], 3))

    for ideposit in range(deposits_array.shape[0]):
        particle_index = int(deposits_array[ideposit, 3])
        if particle_index >= 0:
            particles_array_resized[ideposit] = particles_array[particle_index]

    deposits_array = deposits_array[:, :3]

    return deposits_array, particles_array_resized

def generate_events(padded_size=5000, num_events=10):
    evs = [generate_event() for i in range(num_events)]

    Xs = []
    ys = []
    for X, y in evs:
        X = X[:padded_size]
        y = y[:padded_size]
        X = np.pad(X, ((0, padded_size - X.shape[0]), (0,0)))
        y = np.pad(y, ((0, padded_size - y.shape[0]), (0,0)))
        Xs.append(X)
        ys.append(y)
    X = np.stack(Xs)
    y = np.stack(ys)

    return X, y

def test_graph_mode():
    minibatch_size_in_training = 10

    X,y = generate_events(num_events=100)
    num_samples = X.shape[0]
    num_samples_per_graph = X.shape[1]
    num_features = X.shape[2]

    inputs = tf.keras.Input(shape=(num_samples_per_graph, num_features,), batch_size=minibatch_size_in_training)
    dense = tf.keras.layers.Dense(128, activation=tf.keras.activations.selu)(inputs)
    dense2 = tf.keras.layers.Dense(128, activation=tf.keras.activations.selu)(inputs)

    #compute the distance matrix between points
    #[num_batches, num_samples_per_graph, 128] -> [num_batch, num_samples_per_graph, num_samples_per_graph]
    dm = SparseHashedNNDistance(bin_size=50, max_num_bins=200, num_neighbors=10)(dense)

    #compute the dm*x sparsely across the batches:
    #[num_batches, num_samples_per_graph, num_samples_per_graph] x [num_batches, num_samples_per_graph, 128] -> [num_batch, num_samples_per_graph, 128]
    dm_x_dense = tf.keras.layers.Lambda(lambda args: sparse_dense_matmult_batch(args[0], args[1]))([dm, dense2])

    #just a dummy output for each point [num_batches, num_samples_per_graph, num_features]
    dense_out = tf.keras.layers.Dense(3)(dm_x_dense)

    model = tf.keras.Model(inputs=inputs, outputs=[dense_out, dm], name="test")
    model.compile(loss=["mse", None], optimizer="adam")
    ret = model.fit(X, y, batch_size=minibatch_size_in_training, epochs=10)

    computed_dm = model(X[:1])[1]
    computed_dm = tf.sparse.slice(computed_dm, [0, 0, 0], [1, num_samples_per_graph, num_samples_per_graph])
    computed_dm = tf.sparse.to_dense(computed_dm)[0]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 5))
    plt.imshow(computed_dm, interpolation="none", cmap="binary")
    plt.colorbar()
    plt.title("Learned adjacency matrix")
    plt.savefig("images/dm.png", dpi=300)

    plt.figure(figsize=(5, 5))
    plt.scatter(X[0, :, 1], X[0, :, 2], marker="o", color="red", s=2.0)
    plt.title("Input set (no edges)")
    plt.savefig("images/graph_noedge.png", dpi=300)

    plt.figure(figsize=(5, 5))
    rows, cols = np.where(computed_dm>0)
    edges = np.stack([rows, cols])
    plt.plot(X[0, edges, 1], X[0, edges, 2], linestyle="-", marker="o", color="black", markerfacecolor="red", markeredgecolor="red", markersize=2.0, lw=0.1, alpha=0.2)
    plt.title("Learned graph structure")
    plt.savefig("images/graph.png", dpi=300)

    print("graph mode / keras training successful")

if __name__ == "__main__":
    test_eager()
    test_graph_mode()
    test_timing()