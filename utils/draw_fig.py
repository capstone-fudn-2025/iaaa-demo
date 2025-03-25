import matplotlib

import matplotlib.pyplot as plt

# matplotlib.use('Agg')

def draw_fig(figsize=(15, 5), title="", save_path=None, is_show_fig=False, **kwargs):
    plt.figure(figsize=figsize)
    for key, value in kwargs.items():
        plt.plot(value, label=key)
    plt.legend()
    plt.title(title)
    if save_path is not None:
        plt.savefig(save_path)
    if is_show_fig is True:
        plt.show()
    plt.close()