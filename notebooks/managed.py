from typing import Optional
from jax import numpy as jnp, Array
import numpy as np
from proto import CompartmentContainer, get_cat_indices, StratSpec, CategoryGroup
import pandas as pd
from numbers import Integral, Number
from typing import Optional
from utils import get_category_names


class ManagedIndex:
    def __init__(self, dim, index):
        self.dim = dim
        self.index = index

    def __repr__(self):
        return f"ManagedIndex: maps {self.dim}\n" + repr(self.index)

    def query(self, q):
        if isinstance(self.index, CompartmentContainer):
            qres = self.index.query(q)
            new_subidx, idx_arr = qres, qres.indices
        elif isinstance(self.index, pd.Index):
            pdlookup = pd.Series(index=self.index, data=np.arange(len(self.index)))
            qbackref = pdlookup[q]
            if isinstance(qbackref, Integral):
                qbackref = pdlookup[[q]]
            new_subidx, idx_arr = qbackref.index, np.array(qbackref)
        else:
            raise TypeError(self.index)
        return ManagedIndex(self.dim, new_subidx), _squash_to_slice(idx_arr)


class ManagedCategoryGroupIndex(ManagedIndex):
    def __init__(self, dim: str, index: CategoryGroup):
        super().__init__(dim, index)

    def query(self, q):
        qres = self.index.query(q)
        return ManagedCategoryGroupIndex(self.dim, qres, _squash_to_slice(qres.indices))

    def get_labels(self):
        def label_for_category(category):
            return "_".join(["|".join(strata) for strat, strata in category.traits])

        return [label_for_category(cat) for cat in self.index.categories]

    def __repr__(self):
        return f"ManagedCategoryGroupIndex: maps [{self.dim}]\n" + repr(self.index)


class ManagedArray:
    def __init__(
        self,
        data,
        dims,
        indices: Optional[dict[str, ManagedIndex]] = None,
        labellers=None,
    ):
        if len(dims) != len(data.shape):
            raise ValueError(
                f"Shape mismatch between dims {len(dims)} and data {len(data.shape)}"
            )

        self.data = data
        self.dims = dims

        self._dim_idx = {dim: i for i, dim in enumerate(dims)}
        self.indices = indices or {}
        self.labellers = labellers or {}

    def add_index(self, name, dim, index):
        self.indices[name] = ManagedIndex(dim, index)

    def copy_with(self, **kwargs):
        out_kwargs = kwargs.copy()
        if "data" not in out_kwargs:
            out_kwargs["data"] = self.data.copy()
        if "dims" not in out_kwargs:
            out_kwargs["dims"] = self.dims.copy()
        if "indices" not in out_kwargs:
            out_kwargs["indices"] = self.indices.copy()
        if "labellers" not in out_kwargs:
            out_kwargs["labellers"] = self.labellers.copy()
        return ManagedArray(**out_kwargs)

    def transpose(self, dims):
        if set(dims) != set(self.dims):
            raise Exception("Dimensions must match exactly")
        transposed_data = self.data.transpose([self._dim_idx[d] for d in dims])
        return ManagedArray(transposed_data, dims, self.indices, self.labellers)

    def expand(self, dim, index=-1):
        expanded = jnp.expand_dims(self.data, index)
        if index == -1:
            new_axes = self.dims + [dim]
        else:
            new_axes = []
            for i, a in enumerate(self.dims):
                if i == index:
                    new_axes.append(dim)
                new_axes.append(a)
        return ManagedArray(expanded, new_axes, self.indices, self.labellers)

    def reconcile(self, other):
        s_set = set(self.dims)
        o_set = set(other.dims)
        s_extras = s_set.difference(o_set)
        o_extras = o_set.difference(s_set)
        other_expanded = other
        self_expanded = self
        for eax in list(s_extras):
            other_expanded = other_expanded.expand(eax)
        for eax in list(o_extras):
            self_expanded = self_expanded.expand(eax)
        return self_expanded, other_expanded.transpose(self_expanded.dims)

    def _lop(self, other, op):
        if isinstance(other, Number):
            return self.copy_with(
                data=op(self.data, other)
            )  # , self.dims, self.indices, self.labellers)
        elif isinstance(other, ManagedArray):
            srec, orec = self.reconcile(other)
            return srec.copy_with(data=op(srec.data, orec.data))
            # return ManagedArray(op(srec.data, orec.data), srec.dims, srec.indices, srec.labellers)
        else:
            raise TypeError("Unsupported type", other)

    def _rop(self, other, op):
        if isinstance(other, Number):
            return self.copy_with(data=op(other, self.data))  # , self.axes)
        else:
            raise TypeError("Unsupported type", other)

    def __mul__(self, other):
        return self._lop(other, jnp.multiply)

    def __rmul__(self, other):
        return self._rop(other, jnp.multiply)

    def __add__(self, other):
        return self._lop(other, jnp.add)

    def __radd__(self, other):
        return self._rop(other, jnp.add)

        srec, orec = self.reconcile(other)
        return srec.copy_with(data=srec.data + orec.data)

    def __sub__(self, other):
        return self._lop(other, jnp.subtract)

    def __rsub__(self, other):
        return self._rop(other, jnp.subtract)

        return srec.copy_with(data=srec.data - orec.data)

    def _reduce_dims(self, dims=None):
        if dims is None:
            lifted_ax = None
            reduced_dims = self.dims
        elif isinstance(dims, str):
            lifted_ax = self._dim_idx[dims]
            reduced_dims = [d for d in self.dims if d != dims]
        else:
            lifted_ax = tuple([self._dim_idx[d] for d in dims])
            reduced_dims = [d for d in self.dims if d not in dims]
        return lifted_ax, reduced_dims

    def _liftreduction(self, op, dims=None):
        lifted_ax, reduced_dims = self._reduce_dims(dims)
        reduced_data = getattr(self.data, op)(axis=lifted_ax)

        if len(reduced_dims) == 0:
            return reduced_data
        else:
            reduced_indexers = {
                k: indexer
                for k, indexer in self.indices.items()
                if indexer.dim in reduced_dims
            }

        return self.copy_with(
            data=reduced_data, dims=reduced_dims, indices=reduced_indexers
        )

    @property
    def shape(self):
        return self.data.shape

    def query(self, **kwargs):
        qindices = []
        for idx_name, q in kwargs.items():
            mindex = self.indices[idx_name]
            di = self._dim_idx[mindex.dim]
            new_subidx, qidx = mindex.query(
                q
            )  # self._handle_index_query(mindex.index, q)
            qindices.append((di, (idx_name, new_subidx, qidx)))
        # qindices = sorted(qindices, key=lambda x: x[0])
        qindices = {dimi: q for dimi, q in qindices}
        slicer = []
        new_indices = {}
        for i in range(len(self.dims)):
            if i in qindices:
                idx_name, new_subidx, qidx = qindices[i]
                new_indices[idx_name] = new_subidx
                slicer.append(qidx)
            else:
                slicer.append(...)
        for k, v in self.indices.items():
            if k not in new_indices:
                new_indices[k] = v
        try:
            out_data = self.data[*slicer]
            return ManagedArray(out_data, self.dims, new_indices)
        except:
            raise Exception("Unsupported slice styles; try chaining queries")

    def __repr__(self):
        return (
            f"ManagedArray\n{self.dims} {self.shape}\n"
            + f"Indices:\n{self.indices}\n"
            + f"Data:\n{self.data}"
        )

    def sumcats(self, *args, **kwargs) -> "ManagedArray":
        if (len(args) > 0 and len(kwargs) > 0) or len(args) > 1 or len(kwargs) > 1:
            raise Exception("Only one positional or one kwarg allowed")
        elif len(args) == 1 and len(kwargs) == 0:
            if len(self.indices) == 1:
                idx_name = list(self.indices.keys())[0]
                catgroups = args[0]
            else:
                raise Exception(
                    "Must supply index name in kwarg for multi-index ManagedArray"
                )
        elif len(kwargs) == 1:
            idx_name, catgroups = list(kwargs.items())[0]
        else:
            raise Exception("Unmatched argument types")

        indexer = self.indices[idx_name]
        maps_dim, cat_cmap = indexer.dim, indexer.index
        cat_indices = get_cat_indices(catgroups, cat_cmap)
        # cat_names = get_category_names(catgroups)
        dim_idx = self._dim_idx[maps_dim]

        if len(set([len(c) for c in cat_indices])) == 1:
            slicers = [
                slice(None) if i != dim_idx else np.array(cat_indices)
                for i in range(len(self.dims))
            ]
            # slicers.append(np.array(cat_indices))
            new_data = self.data[*slicers].sum(axis=dim_idx + 1)
        else:
            new_data = jnp.array(
                [
                    self.data[*([slicers] + [c])].sum(axis=dim_idx + 1)
                    for c in cat_indices
                ]
            )

        out_dims = [d if d != maps_dim else "category" for d in self.dims]

        out_indices = {
            name: midx for name, midx in self.indices.items() if midx.dim != maps_dim
        }
        out_indices["category"] = ManagedCategoryGroupIndex("category", catgroups)
        return self.copy_with(data=new_data, dims=out_dims, indices=out_indices)

        if dim_idx == (len(self.dims) - 1):
            slicers = [slice() for i in range(len(self.dims) - 1)]
            if len(set([len(c) for c in cat_indices])) == 1:
                slicers.append(np.array(cat_indices))
                new_data = self.data[*slicers].sum(axis=-1)
            else:
                new_data = jnp.array(
                    [self.data[*([slicers] + [c])].sum(axis=-1) for c in cat_indices]
                )
        else:
            raise Exception("Only timecubes supported currently")
        out_indices = {
            name: midx for name, midx in self.indices.items() if midx.dim != maps_dim
        }
        out_indices["category"] = ManagedCategoryGroupIndex("category", catgroups)
        return ManagedArray(new_data, ["time", "category"], indices=out_indices)

    def sum(self, dims=None, to_dims=None):
        if to_dims is not None:
            if dims is not None:
                raise Exception("Only one of dims and to_dims can be supplied")
            if isinstance(to_dims, str):
                to_dims = [to_dims]
            dims = [d for d in self.dims if d not in to_dims]
        return self._liftreduction("sum", dims=dims)

    def to_pandas_df(self):
        if len(self.dims) > 2:
            raise Exception("Only 2d ManagedArrays supported for Pandas export")

        if len(self.dims) == 1:
            columns = ["data"]
        else:
            data_dim = self.dims[1]

            if data_dim in self.labellers:
                labeller = self.labellers[data_dim]
                columns = labeller(self)
            elif data_dim in self.indices:
                dim_indexer = self.indices[data_dim]
                if hasattr(dim_indexer, "get_labels"):
                    columns = dim_indexer.get_labels()
                else:
                    col_idx = self.indices[data_dim].index
                    if isinstance(col_idx, CompartmentContainer):
                        columns = col_idx.get_labels()
                    else:
                        columns = col_idx
            else:
                columns = None

        return pd.DataFrame(
            index=self.indices["time"].index, data=self.data, columns=columns
        )


def _squash_to_slice(idx_arr):
    # Flat, contiguous
    if (idx_arr[-1] - idx_arr[0]) == (len(idx_arr) - 1):
        if (idx_arr == np.arange(idx_arr[0], idx_arr[-1] + 1)).all():
            return slice(idx_arr[0], idx_arr[-1] + 1)
    # Stepped slice
    diffs = np.diff(idx_arr)
    if len(set(diffs)) == 1:
        step = diffs[0]
        return slice(idx_arr[0], idx_arr[-1] + step, step)

    return idx_arr


class CategoryData(ManagedArray):
    def __init__(self, cats: CategoryGroup, data: Array):
        indexer = ManagedCategoryGroupIndex("category", cats)
        super().__init__(data=data, dims=["category"], indices={"category": indexer})
        self.cats = cats

    def __repr__(self):
        return f"CategoryData:\n{self.cats}\n{self.data}\n"
