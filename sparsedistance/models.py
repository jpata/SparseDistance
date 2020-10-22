import tensorflow as tf
from .utils import split_indices_to_bins, pairwise_dist, sparse_dense_matmult_batch

#Based on the Reformer and GravNet papers
class SparseHashedNNDistance(tf.keras.layers.Layer):
    def __init__(self, max_num_bins=200, bin_size=500, num_neighbors=5, dist_mult=0.1, cosine_dist=False, **kwargs):
        super(SparseHashedNNDistance, self).__init__(**kwargs)
        self.num_neighbors = num_neighbors
        self.dist_mult = dist_mult

        self.cosine_dist = cosine_dist

        #generate the codebook for LSH hashing at model instantiation for up to this many bins
        #set this to a high-enough value at model generation to take into account the largest possible input 
        self.max_num_bins = max_num_bins

        #each bin will receive this many input elements, in total we can accept max_num_bins*bin_size input elements
        #in each bin, we will do a dense top_k evaluation
        self.bin_size = bin_size

    def build(self, input_shape):
        #(n_batch, n_points, n_features)

        #generate the LSH codebook for random rotations
        self.codebook_random_rotations = self.add_weight(
            shape=(input_shape[-1], self.max_num_bins//2), initializer="random_normal", trainable=False, name="lsh_projections"
        )

    @tf.function
    def call(self, inputs, training=True):

        #(n_batch, n_points, n_features)
        point_embedding = inputs

        n_batches = tf.shape(point_embedding)[0]
        n_points = tf.shape(point_embedding)[1]

        #cannot concat sparse tensors directly as that incorrectly destroys the gradient, see
        #https://github.com/tensorflow/tensorflow/blob/df3a3375941b9e920667acfe72fb4c33a8f45503/tensorflow/python/ops/sparse_grad.py#L33
        #therefore, for training, we implement sparse concatenation by hand 
        indices_all = []
        values_all = []

        def func(args):
            ibatch, points_batch = args[0], args[1]
            dm = self.construct_sparse_dm_batch(points_batch)
            inds = tf.concat([tf.expand_dims(tf.cast(ibatch, tf.int64)*tf.ones(tf.shape(dm.indices)[0], dtype=tf.int64), -1), dm.indices], axis=-1)
            vals = dm.values
            return inds, vals

        elems = (tf.range(0, n_batches, delta=1, dtype=tf.int64), point_embedding)
        ret = tf.map_fn(func, elems, fn_output_signature=(tf.int64, tf.float32), parallel_iterations=1)

        shp = tf.shape(ret[0])
        # #now create a new SparseTensor that is a concatenation of the previous ones
        dms = tf.SparseTensor(
            tf.reshape(ret[0], (shp[0]*shp[1], shp[2])),
            tf.reshape(ret[1], (shp[0]*shp[1],)),
            (n_batches, n_points, n_points)
        )

        return tf.sparse.reorder(dms)

    def subpoints_to_sparse_matrix(self, n_points, subindices, subpoints):

        #find the distance matrix between the given points using dense matrix multiplication
        if self.cosine_dist:
            normed = tf.nn.l2_normalize(subpoints, axis=-1)
            dm = tf.linalg.matmul(subpoints, subpoints, transpose_b=True)
        else:
            dm = pairwise_dist(subpoints, subpoints)
            dm = tf.exp(-self.dist_mult*dm)

        dmshape = tf.shape(dm)
        nbins = dmshape[0]
        nelems = dmshape[1]

        #run KNN in the dense distance matrix, accumulate each index pair into a sparse distance matrix
        top_k = tf.nn.top_k(dm, k=self.num_neighbors)
        top_k_vals = tf.reshape(top_k.values, (nbins*nelems, self.num_neighbors))

        indices_gathered = tf.vectorized_map(
            lambda i: tf.gather_nd(subindices, top_k.indices[:, :, i:i+1], batch_dims=1),
            tf.range(self.num_neighbors, dtype=tf.int64))

        indices_gathered = tf.transpose(indices_gathered, [1,2,0])

        #add the neighbors up to a big matrix using dense matrices, then convert to sparse (mainly for testing)
        # sp_sum = tf.zeros((n_points, n_points))
        # for i in range(self.num_neighbors):
        #     dst_ind = indices_gathered[:, :, i] #(nbins, nelems)
        #     dst_ind = tf.reshape(dst_ind, (nbins*nelems, ))
        #     src_ind = tf.reshape(tf.stack(subindices), (nbins*nelems, ))
        #     src_dst_inds = tf.transpose(tf.stack([src_ind, dst_ind]))
        #     sp_sum += tf.scatter_nd(src_dst_inds, top_k_vals[:, i], (n_points, n_points))
        # spt_this = tf.sparse.from_dense(sp_sum)
        # validate that the vectorized ops are doing what we want by hand while debugging
        # dm = np.eye(n_points)
        # for ibin in range(nbins):
        #     for ielem in range(nelems):
        #         idx0 = subindices[ibin][ielem]
        #         for ineigh in range(self.num_neighbors):
        #             idx1 = subindices[ibin][top_k.indices[ibin, ielem, ineigh]]
        #             val = top_k.values[ibin, ielem, ineigh]
        #             dm[idx0, idx1] += val
        # assert(np.all(sp_sum.numpy() == dm))

        #update the output using intermediate sparse matrices, which may result in some inconsistencies from duplicated indices
        sp_sum = tf.sparse.SparseTensor(indices=tf.zeros((0,2), dtype=tf.int64), values=tf.zeros(0, tf.float32), dense_shape=(n_points, n_points))
        for i in range(self.num_neighbors):
           dst_ind = indices_gathered[:, :, i] #(nbins, nelems)
           dst_ind = tf.reshape(dst_ind, (nbins*nelems, ))
           src_ind = tf.reshape(tf.stack(subindices), (nbins*nelems, ))
           src_dst_inds = tf.cast(tf.transpose(tf.stack([src_ind, dst_ind])), dtype=tf.int64)
           sp_sum = tf.sparse.add(
               sp_sum,
               tf.sparse.reorder(tf.sparse.SparseTensor(src_dst_inds, top_k_vals[:, i], (n_points, n_points)))
           )
        spt_this = tf.sparse.reorder(sp_sum)

        return spt_this

    def construct_sparse_dm_batch(self, points):

        #points: (n_points, n_features) input elements for graph construction
        n_points = tf.shape(points)[0]
        n_features = tf.shape(points)[1]

        #compute the number of LSH bins to divide the input points into on the fly
        #n_points must be divisible by bin_size exactly due to the use of reshape
        n_bins = tf.math.floordiv(n_points, self.bin_size)
        #tf.debugging.assert_greater(n_bins, 0)

        #put each input item into a bin defined by the softmax output across the LSH embedding
        mul = tf.linalg.matmul(points, self.codebook_random_rotations[:, :n_bins//2])
        #tf.debugging.assert_greater(tf.shape(mul)[2], 0)

        cmul = tf.concat([mul, -mul], axis=-1)

        #cmul is now an integer in [0..nbins) for each input point
        #bins_split: (n_bins, bin_size) of integer bin indices, which put each input point into a bin of size (n_points/n_bins)
        bins_split = split_indices_to_bins(cmul, n_bins, self.bin_size)

        #parts: (n_bins, bin_size, n_features), the input points divided up into bins
        parts = tf.gather(points, bins_split)

        #sparse_distance_matrix: (n_points, n_points) sparse distance matrix
        #where higher values (closer to 1) are associated with points that are closely related
        sparse_distance_matrix = self.subpoints_to_sparse_matrix(n_points, bins_split, parts)

        return sparse_distance_matrix