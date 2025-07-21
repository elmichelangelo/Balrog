import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import imageio
import os
from natsort import natsorted
from scipy.stats import gaussian_kde
from Handler.helper_classes import MidpointNormalize
import corner
import math


def plot_corner(data_frame, columns, labels, title, ranges=None, show_plot=False, save_plot=False, save_name=None):
    """"""
    data = data_frame[columns].values
    ndim = data.shape[1]

    fig, axes = plt.subplots(ndim, ndim, figsize=(18, 10))

    corner.corner(
        data,
        fig=fig,
        bins=100,
        range=ranges,
        color='#ff8c00',
        smooth=.8,
        smooth1d=.8,
        labels=labels,
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 12},
        hist_kwargs={'alpha': 1},
        scale_hist=True,
        quantiles=[0.16, 0.5, 0.84],
        levels=[0.393, 0.865, 0.989],  # , 0.989
        density=True,
        plot_datapoints=True,
        plot_density=True,
        plot_contours=True,
        fill_contours=True
    )
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#ff8c00', lw=4, label='gaNdalF'),
    ]

    fig.suptitle(f'{title}', fontsize=16)

    fig.legend(handles=legend_elements, loc='upper right', fontsize=12)

    if show_plot is True:
        plt.show()
    if save_plot is True:
        plt.savefig(save_name, dpi=200)
    plt.clf()
    plt.close(fig)


def plot_compare_corner(data_frame_generated, data_frame_true, columns, labels, title, ranges=None, show_plot=False,
                        save_plot=False, save_name=None):

    arr_generated = data_frame_generated[columns].values
    arr_true = data_frame_true[columns].values

    # Quantile für gandalf berechnen
    quantiles_gandalf = np.quantile(arr_generated, q=[0.16, 0.84], axis=0)

    # Quantile für balrog berechnen
    quantiles_balrog = np.quantile(arr_true, q=[0.16, 0.84], axis=0)

    delta_names = ["mean", "median", "q16", "q84"]

    ndim = arr_generated.shape[1]

    fig, axes = plt.subplots(ndim, ndim, figsize=(16, 12))

    # Plot gandalf
    corner.corner(
        arr_generated,
        fig=fig,
        bins=100,
        range=ranges,
        color='#ff8c00',
        smooth=.8,
        smooth1d=.8,
        labels=labels,
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 12},
        hist_kwargs={'alpha': 1},
        scale_hist=False,
        quantiles=[0.16, 0.5, 0.84],
        levels=[0.393, 0.865, 0.989],
        density=True,
        plot_datapoints=True,
        plot_density=True,
        plot_contours=True,
        fill_contours=True
    )

    # Plot balrog
    corner.corner(
        arr_true,
        fig=fig,
        bins=100,
        range=ranges,
        color='#51a6fb',
        smooth=.8,
        smooth1d=.8,
        labels=labels,
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 12},
        hist_kwargs={'alpha': 1},
        scale_hist=False,
        quantiles=[0.16, 0.5, 0.84],
        levels=[0.393, 0.865, 0.989],
        density=True,
        plot_datapoints=True,
        plot_density=True,
        plot_contours=True,
        fill_contours=True
    )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#ff8c00', lw=4, label='gaNdalF'),
        Line2D([0], [0], color='#51a6fb', lw=4, label='Balrog')
    ]

    for i in range(ndim):
        # Titel mit Quantilen manuell hinzufügen
        delta_mean = np.mean(arr_generated[:, i]) - np.mean(arr_true[:, i])
        delta_median = np.median(arr_generated[:, i]) - np.median(arr_true[:, i])
        delta_q16 = quantiles_gandalf[0, i] - quantiles_balrog[0, i]
        delta_q84 = quantiles_gandalf[1, i] - quantiles_balrog[1, i]

        legend_elements.append(Line2D([0], [0], color='#ff8c00', lw=0, label=f'{labels[i]}: mean={delta_mean:.5f}'), )
        legend_elements.append(Line2D([0], [0], color='#ff8c00', lw=0, label=f'{labels[i]}: median={delta_median:.5f}'), )
        legend_elements.append(Line2D([0], [0], color='#ff8c00', lw=0, label=f'{labels[i]}: q16={delta_q16:.5f}'), )
        legend_elements.append(Line2D([0], [0], color='#ff8c00', lw=0, label=f'{labels[i]}: q84={delta_q84:.5f}'), )

    fig.suptitle(f'{title}', fontsize=16)

    fig.legend(handles=legend_elements, loc='upper right', fontsize=12)

    if show_plot is True:
        plt.show()
    if save_plot is True:
        plt.savefig(save_name, dpi=300)
    plt.clf()
    plt.close(fig)


def plot_histo(data_frame, columns, colors=None, bin_size=40, show_plot=False, save_plot=False, save_name=None,
               title=None, log_scale=(False, False), xlabel=""):
    if colors is None:
        colors = ['#fc810a', '#1b78bc', '#9463be']

    for idx, column in enumerate(columns):
        sns.histplot(
            x=data_frame[column].values,
            element="step",
            fill=False,
            color=colors[idx],
            bins=bin_size,
            log_scale=log_scale,
            stat="probability",
            label=f"mock {column}"
        )

    # Set some plot parameters
    plt.title(title)
    plt.xlabel(xlabel)
    plt.legend()
    if save_plot is True:
        plt.savefig(save_name, dpi=300)
    if show_plot is True:
        plt.show()
    plt.clf()


def plot_histo_compare(data_frame_generated, data_frame_true, columns, colors=None, bin_size=40, show_plot=False, save_plot=False,
                       save_name=None, title=None, log_scale=(False, False), xlabel=""):
    """"""

    if colors is None:
        colors = [['#ff8c00', '#51a6fb']]

    for idx, column in enumerate(columns):
        sns.histplot(
            x=data_frame_generated[column].values,
            element="step",
            fill=False,
            color=colors[idx][0],
            bins=bin_size,
            log_scale=log_scale,
            stat="probability",
            label=f"mock {column}"
        )

        if "outside" not in column and "inside" not in column and "BDF_MAG_DERED_CALIB" not in column and "after" not in column:
            sns.histplot(
                x=data_frame_true[column].values,
                element="step",
                fill=False,
                color=colors[idx][1],
                bins=bin_size,
                log_scale=log_scale,
                stat="probability",
                label=f"true {column}"
            )

    # Set some plot parameters
    plt.title(title)
    plt.xlabel(xlabel)
    plt.legend()
    if save_plot is True:
        plt.savefig(save_name, dpi=300)
    if show_plot is True:
        plt.show()
    plt.clf()


def plot_chain(data_frame, plot_name, max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12, columns=None,
               parameter=None, extends=None):
    """
    :param extends: extents={
                "mag r": (17.5, 26),
                "mag i": (17.5, 26),
                "mag z": (17.5, 26),
                "snr": (-11, 55),
                "size ratio": (-1.5, 4),
                "T": (-1, 2.5)
            }
    :param label_font_size:
    :param tick_font_size:
    :param shade_alpha:
    :param max_ticks:
    :param plot_name: "generated observed properties: chat*"
    :param data_frame:
    :param columns: Mutable list, default values are columns = [
            "unsheared/mag_r",
            "unsheared/mag_i",
            "unsheared/mag_z",
            "unsheared/snr",
            "unsheared/size_ratio",
            "unsheared/T"
        ]
    :param parameter: Mutable list, default values are parameter = [
                "mag r",
                "mag i",
                "mag z",
                "snr",              # signal-noise      Range: min=0.3795, max=38924.4662
                "size ratio",       # T/psfrec_T        Range: min=-0.8636, max=4346136.5645
                "T"                 # T=<x^2>+<y^2>     Range: min=-0.6693, max=1430981.5103
            ]
    :return:
    """
    from chainconsumer import ChainConsumer
    df_plot = pd.DataFrame({})

    if columns is None:
        columns = [
            "unsheared/mag_r",
            "unsheared/mag_i",
            "unsheared/mag_z",
            "unsheared/snr",
            "unsheared/size_ratio",
            "unsheared/T"
        ]

    if parameter is None:
        parameter = [
                "mag r",
                "mag i",
                "mag z",
                "snr",              # signal-noise      Range: min=0.3795, max=38924.4662
                "size ratio",       # T/psfrec_T        Range: min=-0.8636, max=4346136.5645
                "T"                 # T=<x^2>+<y^2>     Range: min=-0.6693, max=1430981.5103
            ]

    for col in columns:
        df_plot[col] = np.array(data_frame[col])

    chain = ChainConsumer()
    chain.add_chain(df_plot.to_numpy(), parameters=parameter, name=plot_name)
    chain.configure(
        max_ticks=max_ticks,
        shade_alpha=shade_alpha,
        tick_font_size=tick_font_size,
        label_font_size=label_font_size
    )
    # if extends is not None:
    chain.plotter.plot(
        figsize="page",
        extents=extends
    )
    plt.show()
    plt.clf()


def loss_plot(
        epoch,
        lst_train_loss_per_batch,
        lst_train_loss_per_epoch,
        lst_valid_loss_per_batch,
        lst_valid_loss_per_epoch,
        show_plot,
        save_plot,
        save_name
):
    statistical_figure, ((stat_ax1, stat_ax2), (stat_ax3, stat_ax4)) = plt.subplots(nrows=2, ncols=2)
    statistical_figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.25)
    statistical_figure.suptitle(f"Epoch: {epoch}", fontsize=16)

    # Create dataframe of progress list
    df_training_loss_per_batch = pd.DataFrame({
        "training loss": lst_train_loss_per_batch
    })
    df_training_loss_per_epoch = pd.DataFrame({
        "training loss": lst_train_loss_per_epoch
    })
    df_valid_loss_per_batch = pd.DataFrame({
        "validation loss": lst_valid_loss_per_batch
    })
    df_valid_loss_per_epoch = pd.DataFrame({
        "validation loss": lst_valid_loss_per_epoch
    })

    # Create plot
    df_training_loss_per_batch.plot(
        figsize=(16, 9),
        alpha=0.5,
        marker=".",
        grid=True,
        # yticks=(0, 0.25, 0.5, 0.69, 1.0, 5.0),
        ax=stat_ax1)

    stat_ax1.set_xlabel("batch", fontsize=10, loc='right')
    stat_ax1.set_ylabel("loss", fontsize=12, loc='top')
    stat_ax1.set_title(f"Loss per batch")

    df_training_loss_per_epoch.plot(
        figsize=(16, 9),
        alpha=0.5,
        marker=".",
        grid=True,
        # yticks=(0, 0.25, 0.5, 0.69, 1.0, 5.0),
        ax=stat_ax2)

    stat_ax2.set_xlabel("epoch", fontsize=10, loc='right')
    stat_ax2.set_ylabel("loss", fontsize=12, loc='top')
    stat_ax2.set_title(f"Loss per epoch")

    # Create plot
    df_valid_loss_per_batch.plot(
        figsize=(16, 9),
        alpha=0.5,
        marker=".",
        grid=True,
        # yticks=(0, 0.25, 0.5, 0.69, 1.0, 5.0),
        ax=stat_ax3)

    stat_ax3.set_xlabel("batch", fontsize=10, loc='right')
    stat_ax3.set_ylabel("loss", fontsize=12, loc='top')
    stat_ax3.set_title(f"Loss per batch")

    df_valid_loss_per_epoch.plot(
        figsize=(16, 9),
        alpha=0.5,
        marker=".",
        grid=True,
        # yticks=(0, 0.25, 0.5, 0.69, 1.0, 5.0),
        ax=stat_ax4)

    stat_ax4.set_xlabel("epoch", fontsize=10, loc='right')
    stat_ax4.set_ylabel("loss", fontsize=12, loc='top')
    stat_ax4.set_title(f"Loss per epoch")

    if show_plot is True:
        statistical_figure.show()
    if save_plot is True:
        statistical_figure.savefig(f"{save_name}", dpi=300)

    # Clear and close open figure to avoid memory overload
    statistical_figure.clf()
    plt.close(statistical_figure)
    plt.clf()


def color_color_plot(data_frame_generated, luminosity_type, data_frame_true, colors, show_plot, save_name, extents=None):
    """"""
    from chainconsumer import ChainConsumer
    df_generated_measured = pd.DataFrame({})
    df_true_measured = pd.DataFrame({})
    for color in colors:
        df_generated_measured[f"{color[0]}-{color[1]}"] = \
            np.array(data_frame_generated[f"unsheared/{luminosity_type.lower()}_{color[0]}"]) - np.array(
                data_frame_generated[f"unsheared/{luminosity_type.lower()}_{color[1]}"])
        df_true_measured[f"{color[0]}-{color[1]}"] = \
            np.array(data_frame_true[f"unsheared/{luminosity_type.lower()}_{color[0]}"]) - np.array(
                data_frame_true[f"unsheared/{luminosity_type.lower()}_{color[1]}"])

    arr_true = df_true_measured.to_numpy()
    arr_generated = df_generated_measured.to_numpy()
    parameter = [
        f"unsheared/{luminosity_type.lower()} r-i",
        f"unsheared/{luminosity_type.lower()} i-z"
    ]
    chainchat = ChainConsumer()
    chainchat.add_chain(arr_true, parameters=parameter, name="true observed properties: chat")
    chainchat.add_chain(arr_generated, parameters=parameter,
                        name="generated observed properties: chat*")
    chainchat.configure(max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12)
    chainchat.plotter.plot(
        filename=save_name,
        figsize="page",
        extents=extents
    )
    if show_plot is True:
        plt.show()
    plt.clf()
    plt.close()


def residual_plot(data_frame_generated, data_frame_true, luminosity_type, bands, plot_title, show_plot, save_plot, save_name):
    """"""
    hist_figure, ((stat_ax1), (stat_ax2), (stat_ax3)) = \
        plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
    hist_figure.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
    hist_figure.suptitle(plot_title, fontsize=16)

    lst_axis_res = [
        stat_ax1,
        stat_ax2,
        stat_ax3
    ]

    lst_xlim_res = [
        (-2.5, 2.5),
        (-2.5, 2.5),
        (-2.5, 2.5)
    ]

    df_hist_balrog = pd.DataFrame({
        "dataset": ["balrog" for _ in range(len(data_frame_true[f"unsheared/{luminosity_type.lower()}_r"]))]
    })
    df_hist_generated = pd.DataFrame({
        "dataset": ["generated" for _ in range(len(data_frame_generated[f"unsheared/{luminosity_type.lower()}_r"]))]
    })
    for band in bands:
        df_hist_balrog[f"BDF_{luminosity_type.upper()}_DERED_CALIB - unsheared/{luminosity_type.lower()} {band}"] = data_frame_true[
                                                                              f"BDF_{luminosity_type.upper()}_DERED_CALIB_{band.upper()}"] - \
                                                                          data_frame_true[f"unsheared/{luminosity_type.lower()}_{band}"]
        df_hist_generated[f"BDF_{luminosity_type.upper()}_DERED_CALIB - unsheared/{luminosity_type.lower()} {band}"] = data_frame_generated[
                                                                                 f"BDF_{luminosity_type.upper()}_DERED_CALIB_{band.upper()}"] - \
                                                                             data_frame_generated[
                                                                                 f"unsheared/{luminosity_type.lower()}_{band}"]

    for idx, band in enumerate(bands):
        sns.histplot(
            data=df_hist_balrog,
            x=f"BDF_{luminosity_type.upper()}_DERED_CALIB - unsheared/{luminosity_type.lower()} {band}",
            ax=lst_axis_res[idx],
            element="step",
            stat="density",
            color="dodgerblue",
            bins=50,
            label="balrog"
        )
        sns.histplot(
            data=df_hist_generated,
            x=f"BDF_{luminosity_type.upper()}_DERED_CALIB - unsheared/{luminosity_type.lower()} {band}",
            ax=lst_axis_res[idx],
            element="step",
            stat="density",
            color="darkorange",
            fill=False,
            bins=50,
            label="generated"
        )
        lst_axis_res[idx].axvline(
            x=df_hist_balrog[f"BDF_{luminosity_type.upper()}_DERED_CALIB - unsheared/{luminosity_type.lower()} {band}"].median(),
            color='dodgerblue',
            ls='--',
            lw=1.5,
            label="Mean balrog"
        )
        lst_axis_res[idx].axvline(
            x=df_hist_generated[f"BDF_{luminosity_type.upper()}_DERED_CALIB - unsheared/{luminosity_type.lower()} {band}"].median(),
            color='darkorange',
            ls='--',
            lw=1.5,
            label="Mean generated"
        )
        lst_axis_res[idx].set_xlim(lst_xlim_res[idx][0], lst_xlim_res[idx][1])
        if idx == 0:
            lst_axis_res[idx].legend()
        else:
            lst_axis_res[idx].legend([], [], frameon=False)
    hist_figure.tight_layout()
    if show_plot is True:
        plt.show()

    if save_plot is True:
        plt.savefig(save_name)
    plt.clf()
    plt.close()


def plot_chain_compare(data_frame_generated, data_frame_true, epoch, show_plot, save_name, columns=None, parameter=None,
                       extends=None, max_ticks=5, shade_alpha=0.8, tick_font_size=12, label_font_size=12):
    """"""
    from chainconsumer import ChainConsumer
    if columns is None:
        columns = [
            "unsheared/mag_r",
            "unsheared/mag_i",
            "unsheared/mag_z",
            "unsheared/snr",
            "unsheared/size_ratio",
            "unsheared/T"
        ]

    if parameter is None:
        parameter = [
            "mag r",
            "mag i",
            "mag z",
            "snr",  # signal-noise      Range: min=0.3795, max=38924.4662
            "size ratio",  # T/psfrec_T        Range: min=-0.8636, max=4346136.5645
            "T"  # T=<x^2>+<y^2>     Range: min=-0.6693, max=1430981.5103
        ]

    df_plot_generated = data_frame_generated[columns]
    df_plot_true = data_frame_true[columns]

    chainchat = ChainConsumer()
    chainchat.add_chain(df_plot_true.to_numpy(), parameters=parameter, name="balrog observed properties: chat")
    chainchat.add_chain(df_plot_generated.to_numpy(), parameters=parameter, name="generated observed properties: chat*")
    chainchat.configure(
        max_ticks=max_ticks,
        shade_alpha=shade_alpha,
        tick_font_size=tick_font_size,
        label_font_size=label_font_size
    )
    try:
        chainchat.plotter.plot(
            filename=save_name,
            figsize="page",
            extents=extends
        )
    except:
        print("chain error at epoch", epoch + 1)
    if show_plot is True:
        plt.show()
    plt.clf()


def plot_mean_or_std(data_frame_generated, data_frame_true, lists_to_plot, list_epochs, columns, lst_labels, lst_marker,
                     lst_color, plot_title, show_plot, save_plot, save_name, statistic_type="mean"):
    """"""
    y_label = ""
    for idx_col, col in enumerate(columns):
        if statistic_type.upper() == "MEAN":
            lists_to_plot[idx_col].append(data_frame_generated[col].mean() / data_frame_true[col].mean())
            y_label = "mean(chat*) / mean(chat)"
        elif statistic_type.upper() == "STD":
            lists_to_plot[idx_col].append(data_frame_generated[col].std() / data_frame_true[col].std())
            y_label = "std(chat*) / std(chat)"

    for idx_col, col in enumerate(columns):
        plt.plot(
            list_epochs,
            lists_to_plot[idx_col],
            marker=lst_marker[idx_col],
            linestyle='-',
            color=lst_color[idx_col],
            label=lst_labels[idx_col]
        )
    plt.legend()
    plt.title(plot_title)
    plt.xlabel("epoch")
    plt.ylabel(y_label)

    if show_plot is True:
        plt.show()
    if save_plot is True:
        plt.savefig(save_name, dpi=200)
    plt.clf()
    plt.close()

    return lists_to_plot


def plot_2d_kde(x, y, manual_levels, limits=None, x_label="", y_label="", title="", color=None):
    """"""

    xmin = -2
    xmax = 2
    ymin = -2
    ymax = 2

    if color is None:
        color = "Blues"

    if limits is not None:
        xmin = limits[0]
        xmax = limits[1]
        ymin = limits[2]
        ymax = limits[3]

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)

    f = np.reshape(kernel(positions).T, xx.shape)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Contourf plot
    cfset = ax.contourf(xx, yy, f, cmap=color, alpha=0.3, levels=manual_levels)

    # Or kernel density estimate plot instead of the contourf plot
    # ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])

    # Contour plot
    cset = ax.contour(xx, yy, f, colors='k', levels=manual_levels)

    # Label plot

    ax.clabel(cset, inline=1, fontsize=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(title)


def plot_2d_kde_compare(x1, y1, x2, y2, manual_levels, limits=None, x_label="", y_label="", title="", color=None):
    """"""

    xmin = -2
    xmax = 2
    ymin = -2
    ymax = 2

    if color is None:
        color = "Blues"

    if limits is not None:
        xmin = limits[0]
        xmax = limits[1]
        ymin = limits[2]
        ymax = limits[3]

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values1 = np.vstack([x1, y1])
    kernel1 = gaussian_kde(values1)
    f1 = np.reshape(kernel1(positions).T, xx.shape)

    values2 = np.vstack([x2, y2])
    kernel2 = gaussian_kde(values2)
    f2 = np.reshape(kernel2(positions).T, xx.shape)

    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # manual_levels = np.array([0, 0.8, 1.6, 2.4, 3.2, 4.0, 4.8, 5.6])

    # Contourf plot
    ax.contourf(xx, yy, f1, cmap=color[0], alpha=0.3, levels=manual_levels)
    ax.contourf(xx, yy, f2, cmap=color[1], alpha=0.3, levels=manual_levels)

    # Or kernel density estimate plot instead of the contourf plot
    # ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])

    # Contour plot
    cset1 = ax.contour(xx, yy, f1, colors='k', levels=manual_levels)
    cset2 = ax.contour(xx, yy, f2, colors='k', levels=manual_levels)

    # Label plot
    ax.clabel(cset1, inline=1, fontsize=10)
    ax.clabel(cset2, inline=1, fontsize=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(title)


def make_gif(frame_folder, name_save_folder, fps=10):
    filenames = natsorted(os.listdir(frame_folder))
    images_data = []
    for filename in filenames:
        image = imageio.imread(f"{frame_folder}/{filename}")
        images_data.append(image)
    imageio.mimwrite(uri=f"{name_save_folder}", ims=images_data, format='.gif', duration=int(1000*1/fps))


def plot_balrog_spencer(data_frame):
    import seaborn as sb
    plt.style.use('seaborn')
    sb.set_context("notebook", font_scale=1.5)
    sb.set_style("whitegrid")

    xlim = [15, 26]
    ylim = [-8, 4]

    vmin, vmax = -2.5, 5

    bindx = dict(zip('griz', range(4)))
    b = 'i'
    bi = bindx[b]

    print('Plotting sample...')

    N = 10  # Only plot 10% to run faster

    T_err = data_frame['unsheared/T'] - data_frame['BDF_T']
    T_err = np.log10(np.abs(T_err))

    x = data_frame['BDF_MAG_DERED_CALIB_I'][::N]
    y = data_frame['unsheared/mag_i'][::N] - x

    cmap = 'coolwarm'
    plt.scatter(x, y, s=7, c=T_err[::N], cmap=cmap, norm=MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax))

    # Reference lines
    plt.axhline(-1, lw=3, ls=':', c='k')
    plt.axhline(0, lw=3, ls='--', c='k')
    plt.axhline(1, lw=3, ls=':', c='k')

    plt.xlim(xlim)
    plt.ylim(ylim)

    plt.xlabel('True {}-mag (bdf)'.format(b))
    plt.ylabel('Measured-True {}-mag (cm-bdf)'.format(b))
    cbar = plt.colorbar()
    cbar.set_label('log10(|Meas-True T|)', rotation=270, labelpad=25)
    plt.show()


def plot_bin_offset(data_frame, true_column, measured_column, true_magnitude_percent_deviation=23.25,
                    color="#da7b29", bin_size=0.25, show_plot=False, save_plot=False, save_name="my_plot.png"):
    """"""
    data_frame_copy = data_frame.copy()
    data_frame_copy = data_frame_copy[
        (18 <= data_frame_copy[true_column]) & (data_frame_copy[true_column] < 24.5)]
    true = data_frame_copy[true_column].values
    unsheared_mag_i = data_frame_copy[measured_column].values
    difference = unsheared_mag_i - true

    min_true = np.min(true)
    max_true = np.max(true)
    bins = np.arange(min_true, max_true + bin_size, bin_size)

    bin_stats = {'bin_edges': [], 'std': [], 'mean': [], 'median': [], 'count': []}
    for i in range(len(bins) - 1):
        bin_mask = (true >= bins[i]) & (true < bins[i + 1])
        bin_values = difference[bin_mask]
        bin_stats['bin_edges'].append((bins[i], bins[i + 1]))
        bin_stats['count'].append(np.sum(bin_mask))
        if bin_values.size > 0:
            bin_stats['std'].append(np.std(bin_values))
            bin_stats['mean'].append(np.mean(bin_values))
            bin_stats['median'].append(np.median(bin_values))
        else:
            bin_stats['std'].append(np.nan)
            bin_stats['mean'].append(np.nan)
            bin_stats['median'].append(np.nan)

    weights = np.array(bin_stats['count'])
    mean_values = np.array(bin_stats['mean'])
    median_values = np.array(bin_stats['median'])

    valid_mean_weights = weights[~np.isnan(mean_values)]
    valid_means = mean_values[~np.isnan(mean_values)]
    valid_median_weights = weights[~np.isnan(median_values)]
    valid_median = median_values[~np.isnan(median_values)]

    weighted_mean = np.average(valid_means, weights=valid_mean_weights)
    weighted_median = np.average(valid_median, weights=valid_median_weights)

    percent_deviation_mean = (weighted_mean / true_magnitude_percent_deviation) * 100
    percent_deviation_median = (weighted_median / true_magnitude_percent_deviation) * 100

    df_data = pd.DataFrame({
        "true": true,
        "difference": difference
    })

    # import arviz as az
    # inference_data = az.from_pandas(df_data)

    fig = corner.corner(
        df_data,
        bins=100,
        range=[(17, 25), (-1, 1)],
        color=color,
        smooth=.8,
        smooth1d=.8,
        labels=[true_column, f"{measured_column} - {true_column}"],
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 12},
        hist_kwargs={'alpha': 1},
        scale_hist=True,
        quantiles=[0.16, 0.5, 0.84],
        levels=[0.393, 0.865, 0.989],
        density=True,
        plot_datapoints=True,
        plot_density=True,
        plot_contours=True,
        fill_contours=True
    )
    fig.set_size_inches(18, 10)

    fig.suptitle(
        f'Weighted Mean: {weighted_mean*1000:.4f} mmag, '
        f'Weighted Median: {weighted_median*1000:.4f} mmag, '
        f'Percent Deviation Mean on {true_magnitude_percent_deviation} mag: {percent_deviation_mean:.2f}%, '
        f'Percent Deviation Median on {true_magnitude_percent_deviation} mag: {percent_deviation_median:.2f}%, '
        , fontsize=12)

    ax = fig.axes[2]

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for i in range(len(bin_centers)):
        std = bin_stats['std'][i]
        mean = bin_stats['mean'][i]
        median = bin_stats['median'][i]
        ax.vlines(x=bins[i], ymin=mean - std, ymax=mean + std, color='k', linestyle='-', linewidth=1)
        ax.vlines(x=bins[i + 1], ymin=mean - std, ymax=mean + std, color='k', linestyle='-', linewidth=1)
        ax.hlines(y=mean + std, xmin=bins[i], xmax=bins[i + 1], color='k', linewidth=1)
        ax.hlines(y=mean - std, xmin=bins[i], xmax=bins[i + 1], color='k', linewidth=1)
        ax.hlines(y=mean, xmin=bins[i], xmax=bins[i + 1], color='g', linewidth=1)
        ax.hlines(y=median, xmin=bins[i], xmax=bins[i + 1], color='b', linewidth=1)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax.grid()

    bin_stats['weighted_mean'] = weighted_mean
    bin_stats['weighted_median'] = weighted_median
    bin_stats['percent_deviation_mean'] = percent_deviation_mean
    bin_stats['percent_deviation_median'] = percent_deviation_median

    if show_plot is True:
        print("show plot")
        plt.show()

    if save_plot is True:
        plt.savefig(save_name, dpi=300)

    plt.clf()
    plt.close()

    return bin_stats


def plot_subplots(df, columns, filename, title_prefix="", bins=100, logy=True, ncols=4):
    n_plots = len(columns)
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows))
    axes = axes.flatten()

    for idx, col in enumerate(columns):
        ax = axes[idx]
        sns.histplot(df[col], bins=bins, ax=ax)
        if logy:
            ax.set_yscale("log")
        ax.set_title(f"{title_prefix}{col}")

    for idx in range(n_plots, nrows * ncols):
        fig.delaxes(axes[idx])
    plt.tight_layout()
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close(fig)