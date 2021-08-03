import argparse

import matplotlib.pyplot as plt

from l5kit.environment.monitor_utils import plot_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--foldername', required=True, type=str,
                        help='Foldername of the outputs to plot')
    parser.add_argument('--info_key', type=str,
                        help='The info key to plot')
    args = parser.parse_args()

    log_dir = 'monitor_logs/{}'.format(args.foldername)
    plot_results([log_dir], None, args.info_key)
    plt.savefig('{}/{}'.format(log_dir, args.info_key))
