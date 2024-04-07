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
        T = np.zeros((n, n))

        M, N = self.grid.shape
        for i in range(M):
            for j in range(N):
                current_state = i * N + j
                if self.grid[i, j] == 1:
                    T[current_state, current_state] = 1
                    continue

                neighbors = self.neighbors((i, j))
                if not neighbors:
                    T[current_state, current_state] = 1
                    continue

                for neighbor in neighbors:
                    next_state = neighbor[0] * N + neighbor[1]
                    T[current_state, next_state] = 1 / len(neighbors)

        return T

    def initO(self):
        M, N = self.grid.shape
        num_states = M * N
        O = np.zeros((16, num_states))
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        for i in range(M):
            for j in range(N):
                state_index = i * N + j
                correct_observation = 0
                for k, (di, dj) in enumerate(directions):
                    ni, nj = i + di, j + dj
                    if ni < 0 or ni >= M or nj < 0 or nj >= N or self.grid[ni, nj] == 1:
                        correct_observation |= 1 << (3 - k)
                for obs in range(16):
                    discrepancy = sum((obs >> k) & 1 != (correct_observation >> k) & 1 for k in range(4))
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
        predicted_state = np.dot(alpha, self.trans)
        updated_belief = predicted_state * self.obs[observation]
        return updated_belief

    def backward(self, beta: npt.ArrayLike, observation: int):
        """Perform one iteration of the backward algorithm.
        Args:
          beta (np.ndarray): Current array of probabilities.
          observation (int): Integer representation of bitstring observation.
        Returns:
          np.ndarray: Updated array.
        """
        observation_adjusted = beta * self.obs[observation]
        backward_update = np.dot(observation_adjusted, np.transpose(self.trans))
        return backward_update

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
        alpha = self.init.copy()

        for t, observation in enumerate(observations):
            alpha = self.forward(alpha, observation)
            alphas[t] = alpha

        return alphas, alphas / alphas.sum(axis=1, keepdims=True)

    def smoothing(self, observations: list[int]):
        """
        Executes the smoothing operation over a series of observations to produce refined estimates of state probabilities at each timestep. This method combines information from both past observations (via alpha vectors) and future observations (via beta vectors) to achieve a more accurate state estimation throughout the observation sequence.

        Args:
          observations (list[int]): The series of observed states, each encoded as an integer value.

        Returns:
          np.ndarray: An array containing the beta vectors for each timestep, representing the backward message passing results.
          np.ndarray: An array of smoothed state probabilities for each timestep, offering an enhanced state estimation by amalgamating information from the entire sequence of observations.
        """
        alpha_vectors, _ = self.filtering(observations)
        num_steps = len(observations)
        grid_dim = np.product(self.grid.shape)
        beta_vectors = np.zeros([num_steps, grid_dim])
        beta_vectors[-1] = np.ones(grid_dim) / grid_dim

        for t in range(num_steps - 1, 0, -1):
            beta_vectors[t - 1] = self.backward(beta_vectors[t], observations[t])
        smoothed_probabilities = alpha_vectors * beta_vectors
        normalized_probabilities = (smoothed_probabilities.T / np.sum(smoothed_probabilities, axis=1)).T

        return beta_vectors, normalized_probabilities

    """
    4.4. Parameter learning: Baum-Welch
    """

    def baum_welch(self, observations: list[int]):
        """
        Refines the observation probabilities using the Baum-Welch algorithm. Iteratively updates the
        observation matrix based on the observed sequence until the likelihood change between iterations
        falls below a threshold, indicating convergence.

        Args:
          observations (list[int]): A sequence of integer-encoded observations.

        Returns:
          np.ndarray: The updated observation probabilities matrix.
          list[float]: A list of data likelihoods for each iteration, indicating the model's fit to the data over time.
        """
        epsilon= 1e-10
        M, N = self.grid.shape
        self.obs = np.ones((16, M * N)) / 16
        likelihoods = []
        while True:
            gamma = self.smoothing(observations)[1]
            prob_update = np.zeros([16, self.grid.size])
            for i in range(len(observations)):
                obs = observations[i]
                prob_update[obs] += gamma[i]
            sums = np.sum(prob_update, axis=0, keepdims=True)
            # avoid the situation where the division of zero
            sums[sums == 0] = epsilon
            self.obs = prob_update / sums
            alpha = self.forward(self.init, observations[0])
            score = np.log(np.sum(self.smoothing(observations)[0][0] * alpha))
            likelihoods.append(score)

            if len(likelihoods) > 1 and np.abs(likelihoods[-1] - likelihoods[-2]) < 1e-3:
                break

        return self.obs, likelihoods


