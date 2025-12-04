# qm-evo-cifar-50
Graph-based ψ-evolution classifier achieving ~97% accuracy on CIFAR-10 (animals vs vehicles) using only 50 labeled examples. Includes k-NN, logistic regression baselines, and full reproducible code.
🧬 ψ-Evolution Classifier (CIFAR-10, 50-Label Demo)

This repository contains a minimal, fully reproducible Python script demonstrating a graph-based “ψ-evolution” classifier on CIFAR-10.

The method reaches ~97.4% test accuracy on the animals vs vehicles classification task using only 50 labeled examples (25 per class), using:

no trainable weights

no backpropagation

no neural classifier

Just:

a k-NN graph over embeddings

a normalized Laplacian

a simple reaction–diffusion ψ-update rule

two hyperparameters (α, η)

This is intended as a clear and replicable demonstration of ψ-evo’s label efficiency compared to standard baselines.

📌 Key Result

Using 20k CIFAR-10 train embeddings and only 50 labeled seeds, we obtained:

Method	Train Labels Used	Test Accuracy
k-NN baseline	20,000	0.9755
Logistic regression	20,000	0.9771
k-NN (50 labels)	50	0.9189
Logistic (50 labels)	50	0.9493
ψ-evo (50 labels)	50	0.9740

ψ-evo nearly matches the fully supervised linear probe while using 400× fewer labels.

🔍 What is ψ-Evolution?

ψ-evo is a non-parametric classifier built on a graph:

Extract embeddings (here: ResNet-18 pretrained on ImageNet).

Build a k-NN graph on all train + test embeddings.

Construct the symmetric normalized Laplacian 
𝐿
L.

Pick a small set of labeled seed nodes.

Evolve class amplitude fields ψ using:

𝜓
←
𝜓
−
𝛼
𝐿
𝜓
+
𝜂
 
𝜓
(
1
−
𝜓
)
ψ←ψ−αLψ+ηψ(1−ψ)

Classify test nodes via 
arg
⁡
max
⁡
𝑐
 
𝜓
𝑖
,
𝑐
argmax
c
	​

ψ
i,c
	​

.

No weights.
No training loop.
No gradients.
Just geometry + diffusion.

📦 Files
qm_evo_cifar50.py

Main script that:

Downloads CIFAR-10

Extracts ResNet-18 embeddings (512-dim)

Builds animals vs vehicles labels

Constructs k-NN graph + Laplacian

Samples exactly 50 seeds

Runs:

k-NN (full & 50-label)

Logistic regression (full & 50-label)

ψ-evo (50-label)

Prints final comparison table

▶️ Running the Demo
1. Install dependencies
pip install torch torchvision numpy scipy scikit-learn


(CUDA is optional but recommended.)

2. Run
python qm_evo_cifar50.py


The script will:

download CIFAR-10

extract embeddings

run all baselines + ψ-evo

print a summary table

🚀 ψ-Evo Hyperparameters Used

These settings were found via a small grid search:

k       = 12     # graph neighbors
alpha   = 0.2    # diffusion strength
eta     = 0.03   # nonlinear term weight
n_steps = 60     # evolution steps


These gave the best 50-label performance on this task.

📚 Notes

This is not a competitive CIFAR-10 classifier in the standard deep learning sense — it depends on pretrained embeddings.

ψ-evo is interesting because of its label efficiency, simplicity, and zero-parameter nature.

The method is especially strong when the class boundary aligns with the manifold geometry of the embeddings.

📄 License

This project is released under the MIT License.

🙌 Credits

Developed as part of an experimental exploration of non-parametric, physics-inspired learning dynamics on graphs.
If you use this code or explore ψ-evo further, feel free to open issues or share results.
