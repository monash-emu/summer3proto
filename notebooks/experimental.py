from jax import numpy as jnp


class LA:
    def __init__(self, data, axes):
        self.data = data
        self.axes = axes

        self._ax_to_idx = {k: i for i, k in enumerate(axes)}
        self._idx_to_ax = {v: k for k, v in self._ax_to_idx.items()}

    def transpose(self, axes):
        transposed = self.data.transpose([self._ax_to_idx[a] for a in axes])
        return LA(transposed, axes)

    def expand(self, ax, index=-1):
        expanded = jnp.expand_dims(self.data, index)
        if index == -1:
            new_axes = self.axes + [ax]
        else:
            new_axes = []
            for a, i in enumerate(self.axes):
                if i == index:
                    new_axes.append(ax)
                new_axes.append(a)
        return LA(expanded, new_axes)

    def reconcile(self, other):
        s_set = set(self.axes)
        o_set = set(other.axes)
        s_extras = s_set.difference(o_set)
        o_extras = o_set.difference(s_set)
        other_expanded = other
        self_expanded = self
        for eax in list(s_extras):
            other_expanded = other_expanded.expand(eax)
        for eax in list(o_extras):
            self_expanded = self_expanded.expand(eax)
        return self_expanded, other_expanded.transpose(self_expanded.axes)

    def to_axis(self, ax):
        if ax not in self.axes:
            raise KeyError("Axis not found", ax)
        return [a for a in self.axes if a != ax]

    def __mul__(self, other):
        return self._lop(other, jnp.multiply)

    def __rmul__(self, other):
        return self._rop(other, jnp.multiply)

    def __truediv__(self, other):
        srec, orec = self.reconcile(other)
        return LA(srec.data / orec.data, srec.axes)

    def __rtruediv__(self, other):
        return self._rop(other, jnp.true_divide)

    def _lop(self, other, op):
        if isinstance(other, float):
            return LA(op(self.data, other), self.axes)
        srec, orec = self.reconcile(other)
        return LA(op(srec.data, orec.data), srec.axes)

    def _rop(self, other, op):
        if isinstance(other, float):
            return LA(op(other, self.data), self.axes)
        else:
            raise TypeError("Unsupported type", other)

    def __add__(self, other):
        srec, orec = self.reconcile(other)
        return LA(srec.data + orec.data, srec.axes)

    def __sub__(self, other):
        srec, orec = self.reconcile(other)
        return LA(srec.data - orec.data, srec.axes)

    def sum(self, axis=None):
        return self._liftreduction("sum", axis)

    def _reduce_axes(self, axis=None):
        if axis is None:
            lifted_ax = None
            reduced_axes = self.axes
        elif isinstance(axis, str):
            lifted_ax = self._ax_to_idx[axis]
            reduced_axes = [a for a in self.axes if a != axis]
        else:
            lifted_ax = [self._ax_to_idx[a] for a in axis]
            reduced_axes = [a for a in self.axes if a not in axis]
        return lifted_ax, reduced_axes

    def _liftreduction(self, op, axis=None):
        lifted_ax, reduced_axes = self._reduce_axes(axis)
        return LA(getattr(self.data, op)(axis=lifted_ax), reduced_axes)

    def __repr__(self):
        data_repr = repr(self.data)
        info_repr = f"LA {self.axes} {self.data.shape}\n"
        return info_repr + data_repr
