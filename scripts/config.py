# Path constants
# DATASET_PATH = "../datasets/SWT_Verified_Agentless_Test_Source_Skeleton.json"
# DATASET = "princeton-nlp/SWE-bench_Verified"
DATASET_PATH = "datasets/custom_alias/impacket_issue_1890.json"
DATASET = None
ASSERTFLIP_DIR = "assertflip"
RESULTS_DIR = "test"

# Other constants
max_attempts = 2 # it was previously 10
model = "azure/gpt-4o"
phase_mode = "pass_then_invert"  # Options: pass_then_invert (default mode), direct_fail_variant (for the ablations)
max_generation_retries = 2 #it was previously 10 
