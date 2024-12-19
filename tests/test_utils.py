def get_densities(window_len):
    x = jnp.linspace(0.0, 1.0, window_len, dtype=DTYPE)
    triangle = x * x[::-1]
    return triangle / triangle.sum()