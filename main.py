import argparse
from utils import *

def main():
    parser = argparse.ArgumentParser(
        prog="COMSW4701HW4",
        description="Grid World Localization",
    )
    parser.add_argument("-m", type=int, default=0,
                        help="Mode: 0 for filtering and smoothing (default), "
                             "1 for animating one episode,"
                             "2 for learning observation probabilities")
    parser.add_argument("-t", type=int, default=50, help="number of time steps (default 50)")
    parser.add_argument("-n", type=int, default=500, help="number of episodes for inference (default 500)")
    parser.add_argument("-e", type=float, default=0.0, help="epsilon for animation or learning (default 0)")
    args = parser.parse_args()

    walls = [(0, 4), (0, 10), (0, 14), (1, 0), (1, 1), (1, 4), (1, 6), (1, 7), (1, 9), (1, 11), (1, 13),
             (1, 14), (1, 15), (2, 0), (2, 4), (2, 6), (2, 7), (2, 13), (2, 14), (3, 2), (3, 6), (3, 11)]
    shape = (4, 16)

    if args.m == 1:
        visualize_one_run(shape, walls, args.e, args.t)

    elif args.m == 2:
        obs, likelihoods = learning(shape, walls, args.e, args.t)
        plt.imshow(obs)
        plt.title("Learned observation probabilities")
        plt.xlabel("State vector")
        plt.ylabel("Observation value")
        plt.show()

        plt.plot(likelihoods)
        plt.title("Baum-Welch log likelihood")
        plt.xlabel("Iteration")
        plt.ylabel("Log likelihood")
        plt.show()

    else:
        epsilons = [0.4, 0.2, 0.1, 0.05, 0.02, 0]
        filtering_error, smoothing_error = inference(shape, walls, epsilons, args.t, args.n)

        for e in range(len(epsilons)):
            plt.plot(filtering_error[e], label="e=%.2f" % epsilons[e])
        plt.legend(loc="upper right")
        plt.title("Filtering localization error")
        plt.xlabel("Time Step")
        plt.ylabel("Error")
        plt.show()

        for e in range(len(epsilons)):
            plt.plot(smoothing_error[e], label="e=%.2f" % epsilons[e])
        plt.legend(loc="upper right")
        plt.title("Smoothing localization error")
        plt.xlabel("Time Step")
        plt.ylabel("Error")
        plt.show()


if __name__ == "__main__":
    main()
