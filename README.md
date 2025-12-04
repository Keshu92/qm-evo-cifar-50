<h1>ψ-evo on CIFAR-10 with 50 Labels</h1>

<p>
This repository contains a small but complete experiment on
<strong>CIFAR-10</strong> using a graph–based, “wave-evolution” classifier
we call <strong>ψ-evo</strong>.
We compare it against standard baselines (k-NN and logistic regression)
under two regimes:
</p>

<ul>
  <li><strong>Full labels:</strong> 20k labeled training examples</li>
  <li><strong>Low-label regime:</strong> only <strong>50 labels</strong> (25 per class) for a binary task</li>
</ul>

<p>
The key result: on the <em>animals vs vehicles</em> task with only 50 labels,
ψ-evo gets <strong>97.4% test accuracy</strong>, essentially matching
a fully supervised logistic probe (97.7%) that uses all 20k labels.
</p>

<hr />

<h2>Idea in one paragraph</h2>

<p>
Instead of learning millions of weights, ψ-evo treats the dataset as a graph:
each example is a node, edges connect similar examples,
and class information is encoded as <em>amplitudes</em> on that graph.
Classification is done by iteratively evolving these amplitudes
according to a simple reaction–diffusion update:
</p>

<p style="text-align:center;">
  <img src="https://latex.codecogs.com/png.latex?\psi_{t+1}=\psi_t-\alpha\,L\,\psi_t+\eta\,\psi_t(1-\psi_t)" alt="psi update equation" />
</p>

<ul>
  <li><strong>L</strong> is the (normalized) graph Laplacian built from k-NN similarities on embeddings.</li>
  <li><strong>ψ</strong> is a matrix of class amplitudes over nodes.</li>
  <li><strong>α</strong> controls diffusion (how much label information spreads).</li>
  <li><strong>η</strong> controls a simple nonlinearity that sharpens class regions.</li>
</ul>

<p>
There is <strong>no gradient descent on ψ</strong> and <strong>no learned weights</strong>
inside this update: you initialize ψ using a handful of labeled examples,
evolve it for a fixed number of steps, and then classify each point via
the largest amplitude.
</p>

<hr />

<h2>What’s in this repo?</h2>

<ul>
  <li><code>qm_evo_cifar50.py</code> – main experiment script:
    <ul>
      <li>loads precomputed CIFAR-10 embeddings and labels (or you can point it at your own);</li>
      <li>defines two binary tasks:
        <ul>
          <li><strong>Task 0:</strong> animals vs vehicles</li>
          <li><strong>Task 1:</strong> odd vs even CIFAR class index</li>
        </ul>
      </li>
      <li>builds a k-NN graph and Laplacian;</li>
      <li>runs ψ-evo starting from 50 labeled “seed” nodes;</li>
      <li>evaluates k-NN, logistic regression, and ψ-evo under the same label budget.</li>
    </ul>
  </li>
  <li><code>LICENSE</code> – MIT license (you are free to use/modify with attribution).</li>
</ul>

<p>
If your file name is different, just adjust the command below accordingly.
</p>

<hr />

<h2>Results</h2>

<h3>Task 0 – Animals vs Vehicles</h3>

<p>
Binary label: <code>1 = animal</code>, <code>0 = vehicle</code>, using CIFAR-10 embeddings.
</p>

<h4>Full-label baselines (20k labels, for reference)</h4>

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>Train labels used</th>
      <th>Test accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>k-NN (embeddings)</td>
      <td>20,000</td>
      <td>0.9755</td>
    </tr>
    <tr>
      <td>Logistic regression (linear probe)</td>
      <td>20,000</td>
      <td>0.9771</td>
    </tr>
  </tbody>
</table>

<h4>Low-label regime (only 50 labels total: 25 animals, 25 vehicles)</h4>

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>Train labels used</th>
      <th>Graph / extra structure</th>
      <th>Test accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>k-NN (50-label subset)</td>
      <td>50</td>
      <td>No (just nearest neighbors in embedding space)</td>
      <td>0.9189</td>
    </tr>
    <tr>
      <td>Logistic regression (50 labels)</td>
      <td>50</td>
      <td>No</td>
      <td>0.9493</td>
    </tr>
    <tr>
      <td><strong>ψ-evo (this repo)</strong></td>
      <td>50</td>
      <td>Yes (shared k-NN Laplacian + diffusion)</td>
      <td><strong>0.9740</strong></td>
    </tr>
  </tbody>
</table>

<p>
So with the same 50 labeled examples, ψ-evo essentially matches 
a full-label logistic probe.
</p>

<hr />

<h3>Task 1 – Odd vs Even CIFAR Class Index</h3>

<p>
Here the label is <code>1 = odd CIFAR class index</code>, <code>0 = even index</code> 
(e.g. class 1,3,5,7,9 vs 0,2,4,6,8).
This task is more “semantic-arbitrary”, so performance is lower overall.
</p>

<h4>Full-label baselines (20k labels)</h4>

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>Train labels used</th>
      <th>Test accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>k-NN (embeddings)</td>
      <td>20,000</td>
      <td>0.8203</td>
    </tr>
    <tr>
      <td>Logistic regression (linear probe)</td>
      <td>20,000</td>
      <td>0.8331</td>
    </tr>
  </tbody>
</table>

<h4>Low-label regime (50 labels total: 25 odd, 25 even)</h4>

<table>
  <thead>
    <tr>
      <th>Method</th>
      <th>Train labels used</th>
      <th>Graph / extra structure</th>
      <th>Test accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>k-NN (50-label subset)</td>
      <td>50</td>
      <td>No</td>
      <td>0.6357</td>
    </tr>
    <tr>
      <td>Logistic regression (50 labels)</td>
      <td>50</td>
      <td>No</td>
      <td>0.6564</td>
    </tr>
    <tr>
      <td><strong>ψ-evo (this repo)</strong></td>
      <td>50</td>
      <td>Yes</td>
      <td><strong>0.6762</strong></td>
    </tr>
  </tbody>
</table>

<p>
Here ψ-evo still improves over the 50-label baselines,
but the gap is smaller, reflecting that “odd vs even class index”
is less aligned with the geometry of the embedding space.
</p>

<hr />

<h2>How it works (high level)</h2>

<ol>
  <li><strong>Embeddings</strong>:
    start from fixed 512-dimensional CIFAR-10 embeddings
    (e.g. from a pretrained ResNet or CLIP).
  </li>
  <li><strong>Graph</strong>:
    build a k-nearest neighbor graph over <em>both</em> train and test embeddings
    and compute the normalized graph Laplacian <code>L</code>.
  </li>
  <li><strong>Seeding</strong>:
    choose a small set of labeled train points (e.g. 25 per class)
    and initialize ψ so those seed nodes have amplitude 1.0
    on their class and 0.0 on others.
  </li>
  <li><strong>Evolution</strong>:
    iterate the reaction–diffusion update
    <br />
    <img src="https://latex.codecogs.com/png.latex?\psi_{t+1}=\psi_t-\alpha\,L\,\psi_t+\eta\,\psi_t(1-\psi_t)" alt="psi update equation small" />
    <br />
    for a fixed number of steps, renormalizing rows and clamping seed nodes.
  </li>
  <li><strong>Prediction</strong>:
    after evolution, classify each node by <code>argmax</code> over its ψ row.
  </li>
</ol>

<p>
There are <strong>no trainable parameters inside the ψ-evo dynamics</strong>.
All “learning” comes from:
</p>

<ul>
  <li>the geometry of the embedding space, and</li>
  <li>the small set of labels injected as boundary conditions.</li>
</ul>

<hr />

<h2>Dependencies</h2>

<p>
This repo is intentionally lightweight. You only need:
</p>

<ul>
  <li>Python 3.9+</li>
  <li><code>numpy</code></li>
  <li><code>scipy</code></li>
  <li><code>scikit-learn</code></li>
</ul>

<p>
Optional (if you want to recompute embeddings yourself rather than use
pre-saved <code>.npy</code> files):
</p>

<ul>
  <li><code>torch</code> and <code>torchvision</code> (for CIFAR-10 + ResNet/CLIP)</li>
</ul>

<hr />

<h2>How to run</h2>

<ol>
  <li>Clone the repo:
    <pre><code>git clone https://github.com/&lt;your-user&gt;/qm-evo-cifar-50.git
cd qm-evo-cifar-50
</code></pre>
  </li>
  <li>Install Python dependencies:
    <pre><code>pip install numpy scipy scikit-learn</code></pre>
  </li>
  <li>Make sure the script name in the README matches your file.
      Below we assume it is <code>qm_evo_cifar50.py</code>.</li>
  <li>Run the experiment:
    <pre><code>python qm_evo_cifar50.py</code></pre>
  </li>
</ol>

<p>
You should see printed summaries similar to the tables above.
If you use your own embeddings or different seeds,
numbers will vary a bit, but the relative pattern
(k-NN &lt; logistic &lt; ψ-evo in the 50-label regime)
should be visible.
</p>

<hr />

<h2>Why this might be interesting</h2>

<ul>
  <li><strong>Low parameter count:</strong> the “model” is just a graph Laplacian
      and a simple update rule – no millions of learned weights.</li>
  <li><strong>Physics-flavored view:</strong> classification becomes 
      reaction–diffusion on a graph instead of optimizing a big deep net.</li>
  <li><strong>Label efficiency:</strong> on the animals vs vehicles split,
      50 labels + ψ-evo almost match a 20k-label linear probe.</li>
  <li><strong>Composable:</strong> you can define new binary tasks on top of the same
      embeddings + graph without retraining a deep network.</li>
</ul>

<hr />

<h2>License</h2>

<p>
This project is released under the <strong>MIT License</strong>.
You are free to use, modify, and redistribute the code,
provided you keep the copyright and license notice.
</p>

<hr />

<h2>Acknowledgements</h2>

<p>
This repo grew out of interactive experiments with graph-based diffusion,
semi-supervised learning, and physics-flavored intuitions (Laplacians,
reaction–diffusion, wave-like propagation) applied to modern embeddings.
</p>

