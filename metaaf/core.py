import jax
from jax import jit
import jax.numpy as jnp
from jax.tree_util import Partial, tree_map

from metaaf.optimizer_utils import FeatureContainer


def tree_stack(tree_list, axis):
    """Stack the elements of a list of tree maps with the same structure

    Args:
        tree_list (_type_): List of treemaps
        axis      (_type_): Int axis to stack

    Returns:
        _type_: Single treemap with values stacked.
    """
    leaves, treedef = list(
        zip(*[jax.tree_util.tree_flatten(tree) for tree in tree_list])
    )
    leaves = list(zip(*leaves))
    stacked_leaves = [jnp.stack(l, axis) for l in leaves]
    return treedef[0].unflatten(stacked_leaves)


def tree_slice_axis(tmap, idx_start, idx_len):
    """Get slice of elements in a tree across first k dimensions while preserving others

    Args:
        tmap (_type_): Tree map to slice across objects
        idx_start (_type_): k length list of ints of slice start indices
        idx_len (_type_): k length list of ints of slice lengths

    Returns:
        _type_: Tree map slice where all leaves have been sliced
    """
    n_slice = len(idx_start)

    def slice(x):
        return jax.lax.dynamic_slice(
            x,
            idx_start + [0] * (len(x.shape) - n_slice),
            idx_len + list(x.shape[n_slice:]),
        )

    return tree_map(slice, tmap)


def tree_split(tmap, n_split):
    """Splits all elements in a tree across first dimension

    Args:
        tmap (_type_): Tree map to split across objects
        n_split (_type_): Int num splits

    Returns:
        _type_: Tree map split where all leaves have been split
    """
    return tree_map(
        lambda x: x.reshape([n_split, x.shape[0] // n_split] + list(x.shape[1:])), tmap
    )


def tree_duplicate(tmap, n_dup):
    """Duplicate all elements in a tree across ne first dimension

    Args:
        tmap (_type_): Tree map to split across objects
        n_dup (_type_): Int num duplications

    Returns:
        _type_: Tree map split where all leaves have been duplicated
    """
    return tree_map(lambda x: jnp.stack([x] * n_dup), tmap)


def tree_feature_container(feature, output, data, metadata, key):
    """Take the gradient tree and data dictionary and put them into a featuer container
    object for input to the optimizer

    Args:
        feature (_type_):  Tree map of filter features(usually gradients)
        output (_type_): Dictionary of current filter outputs
        data (_type_): ictionary of current filter inputs
        metadata (_type_): Dictionary of current metadata
        key (_type_): JAX PRNGKey

    Returns:
        _type_: Tree map with filled feature containers
    """
    return tree_map(lambda x: FeatureContainer(x, output, data, metadata, key), feature)


def make_inner_loop(
    outer_fixed,
    meta_loss,
    make_mapped_optmizer,
    get_filter_featues,
    hop_size,
    unroll,
):
    """Function that constructs the inner loop function.

    Args:
        outer_fixed (_type_): Outer kwargs that are fized, this is a dictionary.
        meta_loss (_type_): The meta-loss function
        make_mapped_optmizer (_type_): The function that produces a mapped optimizer
        get_filter_featues (_type_): Filter feature extraction function, typically grad
        hop_size (_type_): The filter hop in samples
        unroll (_type_): The unroll size

    Returns:
        _type_: Function called "run_inner_loop" with signature below
    """

    preprocess = outer_fixed["preprocess"]["preprocess"]
    preprocess_kwargs = outer_fixed["preprocess"]["preprocess_kwargs"]

    postprocess = outer_fixed["postprocess"]["postprocess"]
    postprocess_kwargs = outer_fixed["postprocess"]["postprocess_kwargs"]

    @jit
    def run_inner_loop(
        outer_learnable,
        opt_s,
        filter_s,
        preprocess_s,
        postprocess_s,
        batch_signals_unroll,
        batch_metadata,
        key,
        outer_index,
    ):
        """The actual run inner loop function.

        Args:
            outer_learnable (_type_): Meta-learned parameters
            opt_s (_type_): Optimizer state
            filter_s (_type_): Filter state
            preprocess_s (_type_): Preprocessor state
            postprocess_s (_type_): Postprocessor state
            batch_signals_unroll (_type_): Chunk of signal to process
            batch_metadata (_type_): Any metadata
            key (_type_): JAX PRNGKey
            outer_index (_type_): The index of this loop

        Returns:
            _type_: A tuple where the first element is the final loss, and the second element is a list of all losses, and then a list of all states
        """

        _, opt_update, get_params = make_mapped_optmizer(**outer_learnable)
        n_filter_updates = (
            next(iter(batch_signals_unroll.values())).shape[0] // hop_size
        )

        # innermost update, apply, update, apply loop
        @jit
        def step_update_inner(state, i):
            filter_s, opt_s, preprocess_s, postprocess_s, key = state
            batch_hop = tree_slice_axis(
                batch_signals_unroll, [i * hop_size], [hop_size]
            )

            # run outer trained preprocessing
            key, subkey = jax.random.split(key)
            batch_hop, preprocess_s = preprocess.apply(
                outer_learnable["preprocess_p"],
                preprocess_s,
                subkey,
                data=batch_hop,
                metadata=batch_metadata,
                **preprocess_kwargs
            )

            # run the filter
            key, subkey = jax.random.split(key)
            grad_aux, filter_features = get_filter_featues(
                get_params(opt_s), filter_s, batch_hop, batch_metadata, subkey
            )

            filter_loss, [out, filter_s] = grad_aux

            key, subkey = jax.random.split(key)
            filter_features = tree_feature_container(
                filter_features, out, batch_hop, batch_metadata, subkey
            )

            # run the optimizer update
            opt_s = opt_update(unroll * outer_index + i, filter_features, opt_s)

            # run outer trained postprocessing
            key, subkey = jax.random.split(key)
            out, postprocess_s = postprocess.apply(
                outer_learnable["postprocess_p"],
                postprocess_s,
                subkey,
                data=batch_hop,
                metadata=batch_metadata,
                out=out,
                **postprocess_kwargs
            )

            # make the new state
            state = (filter_s, opt_s, preprocess_s, postprocess_s, key)

            return state, (filter_loss, out)

        # run the sequence of updates
        init_state = (filter_s, opt_s, preprocess_s, postprocess_s, key)
        steps = jnp.arange(n_filter_updates)
        final_state, [all_losses, out] = jax.lax.scan(
            step_update_inner, init_state, steps
        )

        # out = jnp.concatenate(out, 0)
        filter_s, opt_s, preprocess_s, postprocess_s, _ = final_state

        # this is the meta loss function
        final_loss = meta_loss(
            all_losses,
            out,
            batch_signals_unroll,
            batch_metadata,
            outer_index,
            outer_learnable,
        )

        return (
            final_loss,
            (all_losses, out, opt_s, filter_s, preprocess_s, postprocess_s),
        )

    return run_inner_loop


def make_outer_loop(
    outer_fixed,
    meta_loss,
    make_mapped_optmizer,
    meta_opt_update,
    meta_opt_get_params,
    meta_opt_preprocess,
    get_filter_featues,
    unroll,
    hop_size,
):
    """Function to make the outer loop function

    Args:
        outer_fixed (_type_): Meta-learned parameters
        meta_loss (_type_): The meta-loss
        make_mapped_optmizer (_type_): Function to make a mapped optimizer
        meta_opt_update (_type_): Meta optimizer update function
        meta_opt_get_params (_type_): Meta optimizer get params
        meta_opt_preprocess (_type_): Meta optimizer pre-process (e.g. clipping)
        get_filter_featues (_type_): Get filter features function, usually grad
        unroll (_type_): Unroll length
        hop_size (_type_): Filter hop size in samples

    Returns:
        _type_: Returns the outer loop function with signature below
    """
    make_mapped_optmizer = Partial(make_mapped_optmizer, **outer_fixed["optimizer"])

    # learned opt gradient function
    mapped_loss = make_inner_loop(
        outer_fixed=outer_fixed,
        meta_loss=meta_loss,
        make_mapped_optmizer=make_mapped_optmizer,
        get_filter_featues=get_filter_featues,
        hop_size=hop_size,
        unroll=unroll,
    )

    # make the loss batched
    grad_mapped_loss = jax.vmap(
        jax.value_and_grad(mapped_loss, has_aux=True), (None, 0, 0, 0, 0, 0, 0, 0, None)
    )

    @jit
    def batch_run_outer_loop(
        meta_opt_s, filter_p, filter_s, preprocess_s, postprocess_s, batch, key
    ):
        """Run the outer loop and consumes a fulls et of signals

        Args:
            meta_opt_s (_type_): Meta-optimizer state
            filter_p (_type_): Filter parameters
            filter_s (_type_): Filter state
            preprocess_s (_type_): Preprocessor state
            postprocess_s (_type_): Postprocessor state
            batch (_type_): Batch of data
            key (_type_): JAX PRNGKey

        Returns:
            _type_: A tuple with first element being the meta loss, and second element being the filter loss, output, and meta-optimizer state.
        """

        data_shape = next(iter(batch["signals"].values())).shape
        batch_size = data_shape[0]
        n_optimizer_updates = data_shape[1] // (unroll * hop_size)

        # get optimizer params and init in jax format
        outer_learnable = meta_opt_get_params(meta_opt_s)
        opt_init, _, _ = make_mapped_optmizer(**outer_learnable)
        opt_s = jax.vmap(opt_init)(filter_p)

        @jit
        def run_outer_loop(
            opt_s, filter_s, preprocess_s, postprocess_s, meta_opt_s, key
        ):
            def step_update_outer(state, i):
                opt_s, filter_s, preprocess_s, postprocess_s, meta_opt_s, key = state
                outer_learnable = meta_opt_get_params(meta_opt_s)

                batch_signals_unroll = tree_slice_axis(
                    batch["signals"],
                    [0, i * unroll * hop_size],
                    [batch_size, unroll * hop_size],
                )

                key, *subkeys = jax.random.split(key, 1 + batch_size)
                aux, meta_grads = grad_mapped_loss(
                    outer_learnable,
                    opt_s,
                    filter_s,
                    preprocess_s,
                    postprocess_s,
                    batch_signals_unroll,
                    batch["metadata"],
                    jnp.array(subkeys),
                    i,
                )

                cur_meta_loss = aux[0]
                (
                    cur_seq_losses,
                    out,
                    opt_s,
                    filter_s,
                    preprocess_s,
                    postprocess_s,
                ) = aux[1]

                # average, conjugate, and gather the meta gradients
                meta_grads = tree_map(lambda x: jnp.nanmean(x, 0), meta_grads)
                meta_grads = tree_map(lambda x: jnp.conj(x), meta_grads)
                meta_grads = jax.lax.pmean(meta_grads, axis_name="devices")

                # pas the gradients to the meta optimizer
                meta_grads = meta_opt_preprocess(meta_grads)
                meta_opt_s = meta_opt_update(0, meta_grads, meta_opt_s)
                state = (
                    opt_s,
                    filter_s,
                    preprocess_s,
                    postprocess_s,
                    meta_opt_s,
                    key,
                )
                return state, (cur_meta_loss, cur_seq_losses, out)

            # scan over the optimizer updates
            init_state = (
                opt_s,
                filter_s,
                preprocess_s,
                postprocess_s,
                meta_opt_s,
                key,
            )
            steps = jnp.arange(n_optimizer_updates)

            final_state, (meta_losses, filter_loss, filter_out) = jax.lax.scan(
                step_update_outer, init_state, steps
            )

            opt_s, filter_s, preprocess_s, postprocess_s, meta_opt_s, _ = final_state
            return meta_losses, (filter_loss, filter_out, meta_opt_s)

        return run_outer_loop(
            opt_s, filter_s, preprocess_s, postprocess_s, meta_opt_s, key
        )

    return batch_run_outer_loop


def make_online_optimizer(
    outer_learnable, outer_fixed, make_mapped_optmizer, get_filter_featues
):
    """Function to make inference/test-time run function for the optimizer

    Args:
        outer_learnable (_type_): All outer-learned parameters
        outer_fixed (_type_): All outer-learned kwards
        make_mapped_optmizer (_type_): Function to make mapped optimizer
        get_filter_featues (_type_): Filter feature extraction function, typically grad

    Returns:
        _type_: online_step and online_state functions
    """
    optimizer_args = {**outer_learnable, **outer_fixed["optimizer"]}
    opt_init, opt_update, get_params = make_mapped_optmizer(**optimizer_args)

    preprocess = outer_fixed["preprocess"]["preprocess"]
    preprocess_kwargs = outer_fixed["preprocess"]["preprocess_kwargs"]

    postprocess = outer_fixed["postprocess"]["postprocess"]
    postprocess_kwargs = outer_fixed["postprocess"]["postprocess_kwargs"]

    @jit
    def online_state(filter_p):
        """Takes filter parameters and makes optimizer state

        Args:
            filter_p (_type_): Filter parameters dictionary

        Returns:
            _type_: Returns optimizer state
        """
        return opt_init(filter_p)

    @jit
    def online_step(state, batch, key):
        """Takes the state from online_state, a chunk of batch data, and processes it.

        Args:
            state (_type_): State from online_state
            batch (_type_): Slice of signals of size hop_size and signal metadata
            key (_type_): JAX PRNGKey

        Returns:
            _type_: Returns tuple of loss, ouput signals, and new state.
        """
        batch_hop = batch["signals"]
        batch_metadata = batch["metadata"]
        filter_s, opt_s, preprocess_s, postprocess_s, i = state

        # run outer trained preprocessing
        key, subkey = jax.random.split(key)
        batch_hop, preprocess_s = preprocess.apply(
            outer_learnable["preprocess_p"],
            preprocess_s,
            subkey,
            data=batch_hop,
            metadata=batch_metadata,
            **preprocess_kwargs
        )
        # run the filter
        key, subkey = jax.random.split(key)
        aux, filter_features = get_filter_featues(
            get_params(opt_s), filter_s, batch_hop, batch_metadata, subkey
        )
        loss, [out, filter_s] = aux

        # run the optimizer update
        key, subkey = jax.random.split(key)
        filter_features = tree_feature_container(
            filter_features, out, batch_hop, batch_metadata, subkey
        )
        opt_s = opt_update(0, filter_features, opt_s)

        # run outer trained postprocessing
        key, subkey = jax.random.split(key)
        out, postprocess_s = postprocess.apply(
            outer_learnable["postprocess_p"],
            postprocess_s,
            subkey,
            data=batch_hop,
            metadata=batch_metadata,
            out=out,
            **postprocess_kwargs
        )

        return out, loss, (filter_s, opt_s, preprocess_s, postprocess_s, i + 1)

    return online_step, online_state


def make_fit_single(
    outer_learnable,
    outer_fixed,
    infer_meta_loss,
    make_mapped_optmizer,
    get_filter_featues,
    hop_size,
):
    """Makes a function that can fit and process a full signal.

    Args:
        outer_learnable (_type_): All outer-learned parameters
        outer_fixed (_type_): All outer-learned kwargs
        infer_meta_loss (_type_): The inference time meta-loss -- not vectorized
        make_mapped_optmizer (_type_): Function to make mapped optimizer
        get_filter_featues (_type_): Filter feature extraction function, typically grad
        hop_size (_type_): Filter hop size in samples

    Returns:
        _type_: Return a function with signature below
    """
    online_step, online_state = make_online_optimizer(
        outer_learnable=outer_learnable,
        outer_fixed=outer_fixed,
        make_mapped_optmizer=make_mapped_optmizer,
        get_filter_featues=get_filter_featues,
    )

    batch_step = jax.vmap(online_step, (0, 0, 0))

    def fit_single(filter_s, filter_p, preprocess_s, postprocess_s, batch, key):
        """Processes a full length signal

        Args:
            filter_s (_type_): Filter state
            filter_p (_type_): Filter parameters
            preprocess_s (_type_): Preprocessor state
            postprocess_s (_type_): Postprocessor state
            batch (_type_): A full batch of data
            key (_type_): JAX PRNGKey

        Returns:
            _type_: Returns a tuple with first element being the buffered outputs and second elements being the final loss, all losses, any extra outputs, and the final state.
        """

        data_shape = next(iter(batch["signals"].values())).shape
        batch_size = data_shape[0]
        n_filter_updates = data_shape[1] // hop_size

        batch_state = (
            filter_s,
            jax.vmap(online_state)(filter_p),
            preprocess_s,
            postprocess_s,
            jnp.zeros(batch_size),
        )

        all_losses = []
        out = []

        # Iterate over n_steps
        for i in range(n_filter_updates):
            batch_hop_signals = tree_slice_axis(
                batch["signals"], [0, i * hop_size], [batch_size, hop_size]
            )
            batch_hop = {"signals": batch_hop_signals, "metadata": batch["metadata"]}

            key, *subkeys = jax.random.split(key, 1 + batch_size)
            cur_out, loss, batch_state = batch_step(
                batch_state, batch_hop, jnp.array(subkeys)
            )
            all_losses.append(loss)
            out.append(cur_out)

        # we can stack the first output since we know its shape
        # other outputs go in the auxilliary list
        out = tree_stack(out, axis=1)
        first_out = jnp.reshape(
            out["out"] if type(out) is dict else out,
            (batch_size, -1, out["out"].shape[-1]),
        )

        all_losses = jnp.array(all_losses).T
        final_loss = infer_meta_loss(
            all_losses, out, batch["signals"], batch["metadata"], outer_learnable
        )

        aux = (final_loss, all_losses, out, batch_state)
        return first_out, aux

    return fit_single
