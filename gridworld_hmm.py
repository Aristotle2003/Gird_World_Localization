import numpy as np
import numpy.typing as npt


class Gridworld_HMM:
    def __init__(self, size, epsilon: float = 0, walls: list = []):
        if walls:
            self.grid = np.zeros(size)
            for cell in walls:
                self.grid[cell] = 1
        else:
            self.grid = np.random.randint(2, size=size)

        self.init = ((1 - self.grid) / np.sum(self.grid)).flatten('F')

        self.epsilon = epsilon
        self.trans = self.initT()
        self.obs = self.initO()

    def neighbors(self, cell):
        i, j = cell
        M, N = self.grid.shape
        adjacent = [(i, j), (i, j + 1), (i + 1, j), (i, j - 1), (i - 1, j)]
        neighbors = []
        for a1, a2 in adjacent:
            if 0 <= a1 < M and 0 <= a2 < N and self.grid[a1, a2] == 0:
                neighbors.append((a1, a2))
        return neighbors


    """
    4.1 and 4.2. Transition and observation probabilities
    """

    def initT(self):
        n = self.grid.size
        T = np.zeros((n, n))  # Initialize the transition matrix

        M, N = self.grid.shape
        for i in range(M):
            for j in range(N):
                current_state = i * N + j
                if self.grid[i, j] == 1:
                # For walls, the agent stays in the same state
                    T[current_state, current_state] = 1
                    continue

                neighbors = self.neighbors((i, j))  # Calculate neighbors

            # Check if the cell is isolated or incorrectly handled
                if not neighbors:  # If no neighbors, force stay in current state
                    T[current_state, current_state] = 1
                    continue

            # Calculate transition probabilities
                for neighbor in neighbors:
                    next_state = neighbor[0] * N + neighbor[1]
                    T[current_state, next_state] = 1 / len(neighbors)

    # Assert that all rows sum to 1
        assert np.allclose(T.sum(axis=1), 1), "Rows of the transition matrix T do not sum to 1."

        return T





    def initO(self):
        M, N = self.grid.shape
        num_states = M * N
        O = np.zeros((16, num_states))  # Initialize the observation probability matrix

    # Directions corresponding to NESW
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        for i in range(M):
            for j in range(N):
                state_index = i * N + j  # Flatten grid coordinates to a single index
                correct_observation = 0

            # Determine the correct observation by checking adjacent cells
                for k, (di, dj) in enumerate(directions):
                    ni, nj = i + di, j + dj
                # Check if the neighboring cell is blocked or out-of-bounds (treated as blocked)
                    if ni < 0 or ni >= M or nj < 0 or nj >= N or self.grid[ni, nj] == 1:
                        correct_observation |= 1 << (3 - k)

            # Calculate probabilities for all possible observed values
                for obs in range(16):
                    discrepancy = sum((obs >> k) & 1 != (correct_observation >> k) & 1 for k in range(4))
                # Calculate the probability of observing 'obs' given the actual state
                    O[obs, state_index] = ((1 - self.epsilon) ** (4 - discrepancy)) * (self.epsilon ** discrepancy)

        return O



    """
    4.3 Inference: Forward, backward, filtering, smoothing
    """

    def forward(self, alpha: npt.ArrayLike, observation: int):
        """Perform one iteration of the forward algorithm.
        Args:
          alpha (np.ndarray): Current belief state.
          observation (int): Integer representation of bitstring observation.
        Returns:
          np.ndarray: Updated belief state.
        """
        updated_alpha = np.dot(alpha, self.trans) * self.obs[observation]
    # Normalize the updated alpha to ensure it's a probability distribution
        updated_alpha /= np.sum(updated_alpha)
        return updated_alpha


    def backward(self, beta: npt.ArrayLike, observation: int):
        """Perform one iteration of the backward algorithm.
        Args:
          beta (np.ndarray): Current array of probabilities.
          observation (int): Integer representation of bitstring observation.
        Returns:
          np.ndarray: Updated array.
        """
        updated_beta = np.dot(self.trans, self.obs[observation] * beta)
    # Normalize the updated beta
        updated_beta /= np.sum(updated_beta)
        return updated_beta


    def filtering(self, observations: list[int]):
        """Perform filtering over all observations.
        Args:
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Alpha vectors at each timestep.
          np.ndarray: Estimated belief state at each timestep.
        """
        T = len(observations)
        N = self.grid.size
        alphas = np.zeros((T, N))
        alpha = self.init.copy()  # Initial belief state

        for t, observation in enumerate(observations):
            alpha = self.forward(alpha, observation)
            alphas[t] = alpha

        return alphas, alphas / alphas.sum(axis=1, keepdims=True)


    def smoothing(self, observations: list[int]):
        """Perform smoothing over all observations.
        Args:
          observations (list[int]): List of integer observations.
        Returns:
          np.ndarray: Beta vectors at each timestep.
          np.ndarray: Smoothed belief state at each timestep.
        """
        T = len(observations)
        N = self.grid.size
        betas = np.zeros((T, N))
        beta = np.ones(N) / N  # Initialize beta as uniform distribution

    # Run filtering to get all alphas
        alphas, filtered_beliefs = self.filtering(observations)

        for t in range(T-1, -1, -1):
            betas[t] = beta
            if t > 0:
                beta = self.backward(beta, observations[t-1])

        smoothed_beliefs = (alphas * betas) / (alphas * betas).sum(axis=1, keepdims=True)
        return betas, smoothed_beliefs

    """
    4.4. Parameter learning: Baum-Welch
    """

    def baum_welch(self, observations: list[int]):
        M, N = self.grid.shape
        num_states = M * N
        log_likelihoods = []
        convergence_threshold = 1e-3
        converged = False

        while not converged:
            _, gammas = self.smoothing(observations)
            new_obs = np.zeros((16, num_states))

        # Update observation probabilities
            for obs in range(16):
                for state in range(num_states):
                    numerator = sum(gammas[t][state] for t, obs_t in enumerate(observations) if obs_t == obs)
                    denominator = sum(gammas[t][state] for t in range(len(observations)))
                    new_obs[obs, state] = numerator / denominator if denominator > 0 else 0

        # Update the model with new observation probabilities
            self.obs = new_obs

        # Compute the log likelihood for convergence check
            alpha_1 = self.forward(self.init, observations[0])
            log_likelihood = np.log(np.sum(alpha_1 * self.obs[observations[0], :]))
            log_likelihoods.append(log_likelihood)

        # Check for convergence
            if len(log_likelihoods) > 1 and np.abs(log_likelihoods[-1] - log_likelihoods[-2]) <= convergence_threshold:
                converged = True

        return self.obs, log_likelihoods

