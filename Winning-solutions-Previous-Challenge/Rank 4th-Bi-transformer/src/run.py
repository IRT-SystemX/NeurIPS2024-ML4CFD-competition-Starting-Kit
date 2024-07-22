import os
import sys
from my_ingestion import run_model
import time

src_dir = os.path.dirname(os.path.abspath(__file__))        
BENCHMARK_PATH = os.path.join(src_dir, "..", "benchmark.sav")             # Add the path to the saved benchmark file here. If the file does not exist, it will be created there.

def run(model_name):
    model_path = os.path.join(src_dir, "models", model_name)

    print("\n=====================================================")
    print("Starting the model training and evaluation process...")
    print("=====================================================")
    print("start time ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), "\n")
    start = time.time()

    # training & evaluating the model
    run_model(src_dir, model_path, BENCHMARK_PATH)

    end = time.time()
    print("\n===============================================")
    print("Finished! Execution time: ", (end - start)//60, " minutes\n")
    print("===============================================")


if __name__ == "__main__":
    # read input in command line
    model_name_list = sys.argv[1:]
    if len(model_name_list)==0:
        raise ValueError("Provide at least one model name to run the evaluation on e.g. <bi_transformer>")
    
    print(f"Using models: {model_name_list}")
    for model_name in model_name_list:
        if not os.path.exists(os.path.join(src_dir, "models", model_name)):
            raise ValueError(f"Model name {model_name} does not exist, provide a valid model name e.g. <bi_transformer>")
    
    for model_name in model_name_list:
        try:
            run(model_name)
        except:
            print(f"Error in {model_name}, continuing the evaluation with the next model...")
            continue
