# FedPKD
Parameterd Data-Free Knowledge Didtillation for Heterogeneous Federated Learning
# Prepare Dataset:
*To generate non-iid Mnist Dataset following the Dirichlet distribution D(α=0.1) for 20 clients, using 50% of the total available training samples:
<pre><code>cd FedGen/data/Mnist
python generate_niid_dirichlet.py --n_class 10 --sampling_ratio 0.5 --alpha 0.1 --n_user 20
### This will generate a dataset located at FedGen/data/Mnist/u20c10-alpha0.1-ratio0.5/
</code></pre>
-Similarly, to generate non-iid EMnist Dataset, using 10% of the total available training samples:
<pre><code>cd FedGen/data/EMnist
python generate_niid_dirichlet.py --sampling_ratio 0.1 --alpha 0.1 --n_user 20 
### This will generate a dataset located at FedGen/data/EMnist/u20-letters-alpha0.1-ratio0.1/
</code></pre> 
# Run Experiments:
There is a main file "main.py" which allows running all experiments.
```
python main.py --dataset Mnist-alpha0.1-ratio0.5 --algorithm FedAvg --batch_size 32 --num_glob_iters 200 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --personal_learning_rate 0.01 --times 3 
python main.py --dataset Mnist-alpha0.1-ratio0.5 --algorithm FedProx --batch_size 32 --num_glob_iters 200 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --personal_learning_rate 0.01 --times 3 
python main.py --dataset Mnist-alpha0.1-ratio0.5 --algorithm FedPKD --batch_size 32 --num_glob_iters 200 --local_epochs 20 --num_users 10 --lamda 1 --learning_rate 0.01 --model cnn --personal_learning_rate 0.01 --times 3 
```
# main_plot:
For the input attribute algorithms, list the name of algorithms and separate them by comma, e.g. '--algorithms FedAvg,FedProx,FedGen,FedAlign,FedHKD,FedPKD'
```
  python main_plot.py --dataset EMnist-alpha0.1-ratio0.1 --algorithms FedAvg,FedGen --batch_size 32 --local_epochs 20 --num_users 10 --num_glob_iters 200 --plot_legend 1
```
