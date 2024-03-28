import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

from gridworld_hmm import Gridworld_HMM


def loc_error(beliefs, trajectory):
    errors = []
    for i in range(len(trajectory)):
        belief = beliefs[i]
        belief[trajectory[i]] -= 1
        errors.append(np.sum(np.abs(belief)))
    return errors


def inference(shape, walls, epsilons, T, N):
    filtering_error = np.zeros((len(epsilons), T))
    smoothing_error = np.zeros((len(epsilons), T))

    for e in range(len(epsilons)):
        env = Gridworld_HMM(shape, epsilons[e], walls)
        cells = np.nonzero(env.grid == 0)
        indices = cells[0] * env.grid.shape[1] + cells[1]

        for n in range(N):
            trajectory = []
            observations = []
            curr = np.random.choice(indices)
            for t in range(T):
                trajectory.append(np.random.choice(env.trans.shape[0], p=env.trans[curr]))
                curr = trajectory[-1]
                observations.append(np.random.choice(env.obs.shape[0], p=env.obs[:, curr]))
            filtering_error[e] += loc_error(env.filtering(observations)[1], trajectory)
            smoothing_error[e] += loc_error(env.smoothing(observations)[1], trajectory)

    return filtering_error / N, smoothing_error / N


def learning(shape, walls, epsilon, T):
    env = Gridworld_HMM(shape, epsilon, walls)
    cells = np.nonzero(env.grid == 0)
    indices = cells[0] * env.grid.shape[1] + cells[1]

    observations = []
    curr = np.random.choice(indices)
    for t in range(T):
        curr = np.random.choice(env.trans.shape[0], p=env.trans[curr])
        observations.append(np.random.choice(env.obs.shape[0], p=env.obs[:, curr]))

    env.obs = np.ones((16, env.grid.size)) / 16
    learned = env.baum_welch(observations)
    return learned


def visualize_one_run(shape, walls, epsilon, T):
    env = Gridworld_HMM(shape, epsilon, walls)
    cells = np.nonzero(env.grid == 0)
    indices = cells[0] * env.grid.shape[1] + cells[1]

    trajectory = []
    observations = []
    curr = np.random.choice(indices)
    for t in range(T):
        trajectory.append(np.random.choice(env.trans.shape[0], p=env.trans[curr]))
        curr = trajectory[-1]
        observations.append(np.random.choice(env.obs.shape[0], p=env.obs[:, curr]))

    beliefs = env.filtering(observations)[1]
    for i, j in walls:
        beliefs[:, i*shape[1]+j] = -1

    fig, ax = plt.subplots(1, 1)
    cmap = "summer"
    ax.imshow(np.ones(shape), cmap=cmap)
    ax.set_title(f"Estimated distribution with epsilon={epsilon}")
    ax.set_xticks(np.arange(-0.5, shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
    plt.xticks([])
    plt.yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.tight_layout()

    def update(frame):
        ax.clear()
        
        ax.set_title(f"Gridworld HMM with epsilon={epsilon}")
        curr_belief = beliefs[frame].reshape(-1, shape[1])
        ax.imshow(curr_belief, cmap=cmap)
        ax.plot(trajectory[frame] % shape[1], trajectory[frame] // shape[1], 'ro')
        
        ax.set_xticks(np.arange(-0.5, shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, shape[0], 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)
        plt.xticks([])
        plt.yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    
    num_frames = T
    _ = animation.FuncAnimation(fig, update, frames=num_frames, interval=500, repeat=False)

    plt.show()
