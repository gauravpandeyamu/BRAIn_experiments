# BRAIn_experiments
Here, we run experiments to show that the variance of the BRAIn gradient estimate is quite low even when the proposal distribution is away from the Bayesian reward-conditioned posterior (or PPO-optimal policy).

We consider a special case where the reward is uniform for all the samples and hence posterior/ppo-optimal policy $p(y|x, G=1)$ is the same as prior $p(y|x)$.

We consider the following 3 gradient estimators: 
1) BRAIn without baseline. This is similar to DPG gradient estimator (see section 2.3 of Khalifa et al. ICLR 2021)
2) BRAIn without self-normalized baseline. This is similar to GDC++ gradient estimator (See Algorithm 1 of Korbal et al. Neurips 2022)
3) BRAIn gradient estimate

The experiments reveal that BRAIn gradient estimator has a very low variance. The gap with respect to GDC++ and DPG estimators increases as the distance between the proposal and the posterior (or PPO-optimal policy) increases.

In general, in reinforcement learning, the posterior/PPO-optimal policy is intractable and hence, BRAIn will always have a much lower variance than these approaches.

Note that the BRAIn gradient estimate is a biased estimator of KL divergence between the Bayesian reward-conditioned posterior (PPO-optimal policy for BT reward models) and the training policy.
However, it is an unbiased estimator of the self-normalized KL divergence between the same distributions as proved in Theorem 4.2. This novel divergence is $0$ if and only if the posterior equals the policy.

References
 - Khalifa et al. A distributional approach to controllable text generation ICLR 2021

 - Korbak et al. On Reinforcement Learning and Distribution Matching for Fine-Tuning Language Models with no Catastrophic Forgetting, Neurips 2022
