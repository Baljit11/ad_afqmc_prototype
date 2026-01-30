import jax
from jax import tree_util
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .prop.types import PropState


def make_data_mesh() -> Mesh:
    n = jax.local_device_count()
    devices = mesh_utils.create_device_mesh((n,))
    return Mesh(devices, ("data",))


def shard_first_axis(x: jax.Array, mesh: Mesh) -> jax.Array:
    return jax.device_put(x, NamedSharding(mesh, P("data")))


def replicate(x: jax.Array, mesh: Mesh) -> jax.Array:
    return jax.device_put(x, NamedSharding(mesh, P()))


def shard_prop_state(state: PropState, mesh: Mesh | None) -> PropState:
    """
    Shard only (n_walkers,...) leaves, keep global scalars replicated.
    """
    if mesh is None or mesh.size == 1:
        return state

    walkers_sh = tree_util.tree_map(lambda a: shard_first_axis(a, mesh), state.walkers)

    return state._replace(
        walkers=walkers_sh,
        weights=shard_first_axis(state.weights, mesh),
        overlaps=shard_first_axis(state.overlaps, mesh),
        rng_key=replicate(state.rng_key, mesh),
        pop_control_ene_shift=replicate(state.pop_control_ene_shift, mesh),
        e_estimate=replicate(state.e_estimate, mesh),
        node_encounters=replicate(state.node_encounters, mesh),
    )
