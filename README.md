# FedPKD
parameterd Data-Free Knowledge Didtillation for Heterogeneous Federated Learning
# Prepare Dataset:
To generate non-iid Mnist Dataset following the Dirichlet distribution D(α=0.1) for 20 clients, using 50% of the total available training samples:
Similarly, to generate non-iid EMnist Dataset, using 10% of the total available training samples:
# Run Experiments:
There is a main file "main.py" which allows running all experiments.
# main_plot:
For the input attribute algorithms, list the name of algorithms and separate them by comma, e.g. --algorithms FedAvg,FedProx,FedGen,FedAlign,FedHKD,FedPKD
