#!/usr/bin/env python
def calculate_expected_value(alpha, beta, P_S_V, P_N_V, CA, CR, T_I, T_II):
    """
    Calculate the expected value of a decision.

    Arguments:
    alpha -- Probability of a Type I error
    beta -- Probability of a Type II error
    P_S_V -- Probability of a signal given a true state of the world
    P_N_V -- Probability of noise given a true state of the world
    CA -- Utility of a correct accept decision
    CR -- Utility of a correct reject decision
    T_I -- Cost of a Type I error
    T_II -- Cost of a Type II error

    Returns:
    Expected value of the decision.
    """
    EV = ((1 - alpha) * P_S_V * CA) + ((1 - beta) * P_N_V * CR) - (beta * P_N_V * T_II) - (alpha * P_S_V * T_I)
    return EV

# For option A:
alpha = 0.1  # Probability of Type I error (incorrectly rejecting a true hypothesis)
beta = 0.3   # Probability of Type II error (failing to reject a false hypothesis)
P_S_V = 0.8  # Probability of success given the decision is correct
P_N_V = 0.2  # Probability of failure given the decision is correct
CA = 10      # Utility of correct acceptance (completing education)
CR = -5      # Utility of correct rejection (staying in the military)
T_I = -8     # Cost of Type I error
T_II = -2    # Cost of Type II error

EV_A = calculate_expected_value(alpha, beta, P_S_V, P_N_V, CA, CR, T_I, T_II)

# For option B, we will use inverse probabilities for signal and noise as the context has changed:
# Note: The P_S_C and P_N_C values are typically inverses of P_S_V and P_N_V.
P_S_C = 0.3  # Probability of success given the decision is incorrect (staying in the military)
P_N_C = 0.7  # Probability of failure given the decision is incorrect
CA_B = -6    # Utility of incorrect acceptance (not pursuing education further)
CR_B = 5     # Utility of incorrect rejection (missing out on further education)

EV_B = calculate_expected_value(alpha, beta, P_S_C, P_N_C, CA_B, CR_B, T_I, T_II)

print(EV_A)
print(EV_B)
