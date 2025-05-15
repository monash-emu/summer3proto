from proto import *
from jax import lax, jit, grad, make_jaxpr


class NaiveModel:
    def __init__(self, cmap, flows, dyn_params):
        self.cmap = cmap
        self.flows = flows
        self.dyn_params = dyn_params
        self.actual_flows = {}

    def actualize_flows(self):
        for k, v in self.flows.items():
            self.actual_flows[k] = v.actualize(self.cmap)

    def get_runner(self, jit=False):

        self.actualize_flows()

        def run_model(init_state, params, timesteps):
            def state_update(comp_vals, i):
                params["t"] = i
                hdata = self.cmap.wrap_data(comp_vals)
                for k, v in self.dyn_params.items():
                    params[k] = v(hdata, params)

                comp_delta = jnp.zeros_like(comp_vals)
                for k, flow in self.actual_flows.items():
                    flow_vals = flow.get_flow_vals(hdata, params)
                    if hasattr(flow, "src_cmap"):
                        comp_delta = comp_delta.at[flow.src_cmap.indices].subtract(
                            flow_vals
                        )
                    if hasattr(flow, "dest_cmap"):
                        comp_delta = comp_delta.at[flow.dest_cmap.indices].add(
                            flow_vals
                        )
                tstep_data = jnp.clip(hdata.data + comp_delta, 0.0)

                return tstep_data, tstep_data

            final, gathered = lax.scan(
                state_update, init_state, xs=jnp.arange(timesteps)
            )
            return gathered

        if jit:
            run_model = jit(run_model, static_argnames=["timesteps"])

        return run_model
