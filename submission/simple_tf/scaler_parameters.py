## define a function that return the parameters of the scaler
from lips.benchmark.airfransBenchmark import AirfRANSBenchmark

def compute_scaler_parameters(benchmark):
    chunk_sizes=benchmark.train_dataset.get_simulations_sizes()
    no_norm_x=benchmark.train_dataset.get_no_normalization_axis_indices()
    scalerParams={"chunk_sizes":chunk_sizes,"no_norm_x":no_norm_x}
    return scalerParams