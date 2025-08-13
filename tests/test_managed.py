import pytest
import numpy as np
import pandas as pd
from jax import numpy as jnp

from summer3.managed import ManagedIndex, ManagedArray
from summer3.proto import CompartmentContainer, CompartmentMap, Stratification


class TestManagedIndex:
    def test_init(self):
        index = pd.Index(["A", "B", "C"])
        mi = ManagedIndex("test_dim", index)
        assert mi.dim == "test_dim"
        assert mi.index.equals(index)

    def test_repr(self):
        index = pd.Index(["A", "B", "C"])
        mi = ManagedIndex("test_dim", index)
        repr_str = repr(mi)
        assert "ManagedIndex: maps test_dim" in repr_str

    def test_query_with_pandas_index_single_item(self):
        index = pd.Index(["A", "B", "C"])
        mi = ManagedIndex("test_dim", index)

        new_mi, indexer = mi.query(["A"])
        assert new_mi.dim == "test_dim"
        assert new_mi.index.equals(pd.Index(["A"]))
        assert isinstance(indexer, slice) or isinstance(indexer, np.ndarray)

    def test_query_with_pandas_index_multiple_items(self):
        index = pd.Index(["A", "B", "C", "D"])
        mi = ManagedIndex("test_dim", index)

        new_mi, indexer = mi.query(["A", "C"])
        assert new_mi.dim == "test_dim"
        assert new_mi.index.equals(pd.Index(["A", "C"]))

    def test_query_with_compartment_container(self):
        # Create a simple compartment container
        age = Stratification("age", ["0-4", "5-14", "15+"])
        cmap = CompartmentMap.new(age)

        mi = ManagedIndex("test_dim", cmap)

        # Query for specific strata
        new_mi, indexer = mi.query([(age, ["0-4"])])
        assert new_mi.dim == "test_dim"
        assert len(new_mi.index.compartments) == 1

    def test_query_invalid_index_type(self):
        mi = ManagedIndex("test_dim", "invalid_index")
        with pytest.raises(TypeError):
            mi.query(["A"])


class TestManagedArray:
    def test_init_basic(self):
        data = jnp.array([[1, 2, 3], [4, 5, 6]])
        dims = ["time", "compartment"]
        ma = ManagedArray(data, dims)

        assert jnp.array_equal(ma.data, data)
        assert ma.dims == dims
        assert ma.indices == {}
        assert ma.labellers == {}

    def test_init_with_shape_mismatch(self):
        data = jnp.array([[1, 2, 3], [4, 5, 6]])  # 2D array
        dims = ["time"]  # Only 1 dimension

        with pytest.raises(ValueError, match="Shape mismatch"):
            ManagedArray(data, dims)

    def test_add_index(self):
        data = jnp.array([1, 2, 3])
        dims = ["compartment"]
        ma = ManagedArray(data, dims)

        index = pd.Index(["A", "B", "C"])
        ma.add_index("comp_idx", "compartment", index)

        assert "comp_idx" in ma.indices
        assert ma.indices["comp_idx"].dim == "compartment"
        assert ma.indices["comp_idx"].index.equals(index)

    def test_copy_with_defaults(self):
        data = jnp.array([1, 2, 3])
        dims = ["compartment"]
        indices = {"test": ManagedIndex("compartment", pd.Index(["A", "B", "C"]))}
        ma = ManagedArray(data, dims, indices)

        copy_ma = ma.copy_with()

        assert jnp.array_equal(copy_ma.data, ma.data)
        assert copy_ma.dims == ma.dims
        assert copy_ma.indices == ma.indices

    def test_copy_with_override(self):
        data = jnp.array([1, 2, 3])
        dims = ["compartment"]
        ma = ManagedArray(data, dims)

        new_data = jnp.array([4, 5, 6])
        copy_ma = ma.copy_with(data=new_data)

        assert jnp.array_equal(copy_ma.data, new_data)
        assert copy_ma.dims == ma.dims

    def test_shape_property(self):
        data = jnp.array([[1, 2, 3], [4, 5, 6]])
        dims = ["time", "compartment"]
        ma = ManagedArray(data, dims)

        assert ma.shape == (2, 3)

    def test_simplify(self):
        data = jnp.array([[[1]], [[2]]])  # Shape (2, 1, 1)
        dims = ["time", "age", "location"]
        ma = ManagedArray(data, dims)

        simplified = ma.simplify()

        assert simplified.shape == (2,)
        assert simplified.dims == ["time"]

    def test_transpose(self):
        data = jnp.array([[1, 2, 3], [4, 5, 6]])
        dims = ["time", "compartment"]
        ma = ManagedArray(data, dims)

        transposed = ma.transpose(["compartment", "time"])

        assert transposed.shape == (3, 2)
        assert transposed.dims == ["compartment", "time"]

    def test_transpose_dimension_mismatch(self):
        data = jnp.array([[1, 2, 3], [4, 5, 6]])
        dims = ["time", "compartment"]
        ma = ManagedArray(data, dims)

        with pytest.raises(Exception, match="Dimensions must match exactly"):
            ma.transpose(["time", "age"])

    def test_expand_at_end(self):
        data = jnp.array([1, 2, 3])
        dims = ["compartment"]
        ma = ManagedArray(data, dims)

        expanded = ma.expand("age")

        assert expanded.shape == (3, 1)
        assert expanded.dims == ["compartment", "age"]

    def test_expand_at_index(self):
        data = jnp.array([1, 2, 3])
        dims = ["compartment"]
        ma = ManagedArray(data, dims)

        expanded = ma.expand("age", index=0)

        assert expanded.shape == (1, 3)
        assert expanded.dims == ["age", "compartment"]

    def test_reconcile(self):
        data1 = jnp.array([1, 2, 3])
        ma1 = ManagedArray(data1, ["compartment"])

        data2 = jnp.array([4, 5])
        ma2 = ManagedArray(data2, ["age"])

        ma1_rec, ma2_rec = ma1.reconcile(ma2)

        assert set(ma1_rec.dims) == {"compartment", "age"}
        assert set(ma2_rec.dims) == {"compartment", "age"}

    def test_arithmetic_with_number(self):
        data = jnp.array([1, 2, 3])
        dims = ["compartment"]
        ma = ManagedArray(data, dims)

        # Test multiplication
        result = ma * 2
        assert jnp.array_equal(result.data, jnp.array([2, 4, 6]))

        # Test addition
        result = ma + 1
        assert jnp.array_equal(result.data, jnp.array([2, 3, 4]))

        # Test subtraction
        result = ma - 1
        assert jnp.array_equal(result.data, jnp.array([0, 1, 2]))

    def test_arithmetic_with_managed_array(self):
        data1 = jnp.array([1, 2, 3])
        ma1 = ManagedArray(data1, ["compartment"])

        data2 = jnp.array([2, 3, 4])
        ma2 = ManagedArray(data2, ["compartment"])

        result = ma1 + ma2
        assert jnp.array_equal(result.data, jnp.array([3, 5, 7]))

    def test_reverse_arithmetic(self):
        data = jnp.array([1, 2, 3])
        dims = ["compartment"]
        ma = ManagedArray(data, dims)

        result = 10 - ma
        assert jnp.array_equal(result.data, jnp.array([9, 8, 7]))

    def test_sum_all_dims(self):
        data = jnp.array([[1, 2, 3], [4, 5, 6]])
        dims = ["time", "compartment"]
        ma = ManagedArray(data, dims)

        result = ma.sum()
        assert result == 21  # Sum of all elements

    def test_sum_specific_dim(self):
        data = jnp.array([[1, 2, 3], [4, 5, 6]])
        dims = ["time", "compartment"]
        ma = ManagedArray(data, dims)

        result = ma.sum("time")
        assert jnp.array_equal(result.data, jnp.array([5, 7, 9]))
        assert result.dims == ["compartment"]

    def test_sum_to_dims(self):
        data = jnp.array([[1, 2, 3], [4, 5, 6]])
        dims = ["time", "compartment"]
        ma = ManagedArray(data, dims)

        result = ma.sum(to_dims="compartment")
        assert jnp.array_equal(result.data, jnp.array([5, 7, 9]))
        assert result.dims == ["compartment"]

    def test_query_single_index_single_arg(self):
        data = jnp.array([1, 2, 3])
        dims = ["compartment"]
        index = pd.Index(["A", "B", "C"])
        ma = ManagedArray(data, dims)
        ma.add_index("comp_idx", "compartment", index)

        result = ma.query(["A"])
        assert result.shape == (1,)
        assert jnp.array_equal(result.data, jnp.array([1]))

    def test_query_with_kwargs(self):
        data = jnp.array([1, 2, 3])
        dims = ["compartment"]
        index = pd.Index(["A", "B", "C"])
        ma = ManagedArray(data, dims)
        ma.add_index("comp_idx", "compartment", index)

        result = ma.query(comp_idx=["A", "C"])
        assert result.shape == (2,)
        assert jnp.array_equal(result.data, jnp.array([1, 3]))

    def test_query_invalid_args(self):
        data = jnp.array([1, 2, 3])
        dims = ["compartment"]
        ma = ManagedArray(data, dims)

        with pytest.raises(Exception):
            ma.query(["A"], comp_idx=["B"])  # Both args and kwargs

    def test_to_pandas_df_1d(self):
        data = jnp.array([1, 2, 3])
        dims = ["time"]
        time_index = pd.Index([0, 1, 2])
        ma = ManagedArray(data, dims)
        ma.add_index("time", "time", time_index)

        df = ma.to_pandas_df()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 1)
        assert list(df.columns) == ["data"]

    def test_to_pandas_df_2d(self):
        data = jnp.array([[1, 2], [3, 4], [5, 6]])
        dims = ["time", "compartment"]
        time_index = pd.Index([0, 1, 2])
        comp_index = pd.Index(["A", "B"])
        ma = ManagedArray(data, dims)
        ma.add_index("time", "time", time_index)
        ma.add_index("compartment", "compartment", comp_index)

        df = ma.to_pandas_df()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)

    def test_to_pandas_df_unsupported_dims(self):
        data = jnp.array([[[1, 2]], [[3, 4]]])
        dims = ["time", "age", "compartment"]
        ma = ManagedArray(data, dims)

        with pytest.raises(Exception, match="Only 2d ManagedArrays supported"):
            ma.to_pandas_df()

    def test_query_single_day_with_date_string(self):
        # Create test data with time dimension
        data = jnp.array([10, 20, 30, 40, 50])
        dims = ["time"]
        ma = ManagedArray(data, dims)
        
        # Create a date index with string representations
        date_strings = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]
        date_index = pd.Index(date_strings)
        ma.add_index("date", "time", date_index)
        
        # Query for a single day using string representation
        result = ma.query(date=["2024-01-03"])
        
        # Verify the result
        assert result.shape == (1,)
        assert jnp.array_equal(result.data, jnp.array([30]))
        assert result.dims == ["time"]

    def test_rolling_equivalence_with_pandas(self):
        """Test that ManagedArray.rolling performs equivalently to pandas DataFrame.rolling."""
        # Create test data - a simple time series
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        dims = ["time"]
        ma = ManagedArray(data, dims)
        
        # Create a pandas DataFrame with the same data
        df = pd.DataFrame({"value": np.array(data)})
        
        # Test rolling sum with window=3
        window = 3
        ma_rolling_sum = ma.rolling(window, jnp.sum)
        df_rolling_sum = df.rolling(window).sum()
        
        # Compare results - pandas puts NaN for the first (window-1) values
        # Our implementation should do the same
        expected = df_rolling_sum["value"].values
        actual = np.array(ma_rolling_sum.data)
        
        # Check that NaN values match in the first (window-1) positions
        assert np.isnan(actual[0]) and np.isnan(expected[0])
        assert np.isnan(actual[1]) and np.isnan(expected[1])
        
        # Check that the rolling sums match for valid positions
        np.testing.assert_array_almost_equal(actual[window-1:], expected[window-1:])
        
        # Test rolling mean with window=4
        window = 4
        ma_rolling_mean = ma.rolling(window, jnp.mean)
        df_rolling_mean = df.rolling(window).mean()
        
        expected_mean = df_rolling_mean["value"].values
        actual_mean = np.array(ma_rolling_mean.data)
        
        # Check NaN values for first (window-1) positions
        for i in range(window-1):
            assert np.isnan(actual_mean[i]) and np.isnan(expected_mean[i])
        
        # Check that the rolling means match for valid positions
        np.testing.assert_array_almost_equal(actual_mean[window-1:], expected_mean[window-1:])
        
        # Test that dimensions and indices are preserved
        assert ma_rolling_sum.dims == ma.dims
        assert ma_rolling_sum.shape == ma.shape
        assert ma_rolling_sum.indices == ma.indices

    def test_repr(self):
        data = jnp.array([1, 2, 3])
        dims = ["compartment"]
        ma = ManagedArray(data, dims)

        repr_str = repr(ma)
        assert "ManagedArray" in repr_str
        assert "compartment" in repr_str
