import numpy as np
import matplotlib.pyplot as plt

# Constants
true_theta = 0
policy_theta = 1  
proposal_theta = 1

num_samples = 8
num_iterations = 2000
sigma_squared = 1
true_gradient = true_theta - policy_theta
# Compute base model probabilities
def compute_p_x(samples):
    p_x = np.exp(-0.5 * (samples - true_theta)**2) / np.sqrt(2 * np.pi)
    return p_x
 
# Compute policy q_theta probabilities
def compute_q_theta_x(samples):
    q_theta_x = np.exp(-0.5 * (samples - policy_theta)**2) / np.sqrt(2 * np.pi)
    return q_theta_x

# Sample from proposal distribution
def sample_from_proposal(proposal_theta, num_samples):
    samples = np.random.normal(proposal_theta, np.sqrt(sigma_squared), num_samples)
    return samples

# Compute the probability assigned by proposal
def compute_proposal_x(proposal_theta, samples):
    proposal = np.exp(-0.5 * (samples - proposal_theta)**2) / np.sqrt(2 * np.pi)
    return proposal

# Gradient of log q_theta wrt policy_theta
def compute_gradients(samples, policy_theta):
    return (samples - policy_theta) / sigma_squared

# Compute KL divergence between proposal and p_x
def compute_kl_divergence(proposal_theta, true_theta=true_theta, sigma_squared=1):
    kl_divergence = 0.5 * ((proposal_theta - true_theta) ** 2) / sigma_squared
    return kl_divergence

# DPG gradient estimator (see section 2.3 of Khalifa et al. ICLR 2021)
def estimator_dpg(proposal_theta):
    gradients = []
    for _ in range(num_iterations):
        samples = sample_from_proposal(proposal_theta, num_samples)
        p_x = compute_p_x(samples)
        proposal_x = compute_proposal_x(proposal_theta, samples)
        weights = (1/num_samples) * (p_x / proposal_x)
        grad = compute_gradients(samples, policy_theta)
        weighted_grad = grad * weights
        gradients.append(np.sum(weighted_grad))
    variance = np.var(gradients)
    return variance

# GDC++ gradient estimator (See Algorithm 1 of Korbal et al. Neurips 2022)
def estimator_gdcpp(proposal_theta):
    gradients = []
    for _ in range(num_iterations):
        samples = sample_from_proposal(proposal_theta, num_samples)
        p_x = compute_p_x(samples)
        q_theta_x = compute_q_theta_x(samples)
        proposal_x = compute_proposal_x(proposal_theta, samples)
        weights = (1/num_samples) * (p_x / proposal_x - q_theta_x/proposal_x)
        grad = compute_gradients(samples, policy_theta)
        weighted_grad = grad * weights
        gradients.append(np.sum(weighted_grad))
    variance = np.var(gradients)
    return variance

# BRAIn gradient estimator
def estimator_brain(proposal_theta):
    gradients = []
    for _ in range(num_iterations):
        samples = sample_from_proposal(proposal_theta, num_samples)
        p_x = compute_p_x(samples)
        q_theta_x = compute_q_theta_x(samples)
        proposal_x = compute_proposal_x(proposal_theta, samples)
        alpha = p_x / proposal_x
        alpha_hat = alpha/alpha.sum()
        beta = q_theta_x/proposal_x
        beta_hat = beta/beta.sum()
        weights = alpha_hat - beta_hat
        grad = compute_gradients(samples, policy_theta)
        weighted_grad = grad * weights
        gradients.append(np.sum(weighted_grad))
    variance = np.var(gradients)
    return variance

# Compute variance for a range of proposal means
proposal_means = np.linspace(0, 1, 20)
kl_divergences = [compute_kl_divergence(pm) for pm in proposal_means]
variances_dpg = [estimator_dpg(pm) for pm in proposal_means]
variances_gdcpp = [estimator_gdcpp(pm) for pm in proposal_means]
variances_brain = [estimator_brain(pm) for pm in proposal_means]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(kl_divergences, variances_dpg, label='DPG')
plt.plot(kl_divergences, variances_gdcpp, label='GDCPP')
plt.plot(kl_divergences, variances_brain, label='Brain')
plt.xlabel('KL Divergence (proposal || p_x)')
plt.ylabel('Variance')
plt.legend()
plt.title('KL Divergence vs. Variance')
plt.grid(True)

# Save the graphs
plt.savefig('proposal_mean_vs_variance.png')

print("Graphs have been saved.")
