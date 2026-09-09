# Paper equations and implementation

This file is the source-of-truth map between the manuscript and released code.
Images are stored in `[0,1]` and mapped to `[-1,1]` before FM training; generated
model outputs are mapped back to `[0,1]` before reward scoring and export.

## Conditional Flow Matching

For independent Gaussian noise \(x_0\), a real endpoint \(x_1\), and
\(t\sim U(0,1)\), the affine CondOT path and target velocity are

\[
x_t=(1-t)x_0+t x_1,\qquad u_t=x_1-x_0.
\]

The base generator minimizes

\[
\mathcal L_{\mathrm{FM}}
=\mathbb E\|v_\theta(x_t,t,c,m)-u_t\|_2^2.
\]

Implemented by `AffineProbPath(CondOTScheduler())` and
`compute_fm_mse_loss` in `utils/utils_fm.py`.

Sampling integrates \(dx_t/dt=v_\theta(x_t,t,c,m)\). The reported pools use
explicit Euler with ten uniform steps (`solver_args` in both YAML configs).

## Classifier rewards

The frozen judge probabilities are the mean softmax probabilities of three
ResNet-18 models. Let \(p_y\) be the probability of the conditioning class.
`rewards.reward_from_probs` implements:

\[
r_{\mathrm{soft}}=p_y,
\]

\[
r_{\mathrm{focal}}=(1-p_y)^\gamma p_y,\qquad \gamma=2,
\]

\[
r_{\mathrm{util}}=
\begin{cases}
(1-p_y)^\gamma p_y,&\arg\max_k p_k=y,\\
-\eta,&\text{otherwise},
\end{cases}
\qquad \eta=1.
\]

The command-line name for Soft-RL is `REWARD_TYPE=prob`.

## Group selection and self-distillation

Each condition is repeated \(G=4\) times and paired with independent Gaussian
noise, producing four rollouts. For the \(k=2\) highest raw rewards,
`group_topk_weights` in `utils/rl_objectives.py` assigns

\[
w_i=G/k=2
\]

and assigns zero to the remaining rollouts, so each group’s weights sum to
\(G\).

The selected rollout \(\hat x_1^{(i)}\) becomes a detached pseudo-target with
\(\hat u_t^{(i)}=\hat x_1^{(i)}-x_0^{(i)}\). The implemented loss is

\[
\mathcal L_{\mathrm{distill}}
=\mathbb E\left[\frac1G\sum_i w_i
\|v_\theta-\hat u_t^{(i)}\|_2^2\right]
+\lambda_{\mathrm{ref}}\mathbb E\|v_\theta-v_{\theta_0}\|_2^2,
\]

\[
\mathcal L=(1-\lambda_{\mathrm{real}})\mathcal L_{\mathrm{distill}}
+\lambda_{\mathrm{real}}\mathcal L_{\mathrm{FM}}^{\mathrm{real}},
\]

with \(\lambda_{\mathrm{real}}=0.3\) and
\(\lambda_{\mathrm{ref}}=0.05\). `rl_trainier.py` implements these terms in
`_adv_weighted_flow_matching_loss` and `training_step`.

## Fixed reported settings

| Setting | Value |
|---|---:|
| Base FM epochs | 200 |
| Base FM learning rate | \(10^{-4}\) |
| Base FM effective batch | 256 |
| RL updates | 4,000 |
| RL learning rate | \(10^{-5}\) |
| Group size \(G\) | 4 |
| Survivors \(k\) | 2 |
| Euler steps | 10 |
| Gradient clip | 1.0 |
| \(\lambda_{\mathrm{real}}\) | 0.3 |
| \(\lambda_{\mathrm{ref}}\) | 0.05 |
| Checkpoint interval | 1,000 updates |

RL checkpoint selection is a separate development-only probing step described
in `CHECKPOINT_SELECTION.md`.
