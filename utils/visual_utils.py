import logging
import os
import warnings
import networkx as nx
import pandas as pd
import numpy as np
import torch

logging.getLogger('matplotlib').setLevel(logging.WARNING)
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

warnings.filterwarnings("ignore", category=matplotlib.cbook.MatplotlibDeprecationWarning)


options = {
    'node_size': 2,
    'edge_color': 'black',
    'linewidths': 1,
    'width': 0.5
}

CMAP = cm.jet


def plot_graphs_list_new(graphs, title='title', rows=1, cols=1, save_dir=None):
    batch_size = len(graphs)
    max_num = min(batch_size, rows * cols)
    figure = plt.figure(figsize=(cols, rows))

    for i in range(max_num):
        idx = i * (batch_size // max_num)
        if not isinstance(graphs[idx], nx.Graph):
            G = graphs[idx].g.copy()
        else:
            G = graphs[idx].copy()
        assert isinstance(G, nx.Graph)
        G.remove_nodes_from(list(nx.isolates(G)))
        ax = plt.subplot(rows, cols, i + 1)
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=False, **options)
        #         ax.grid(False)
        ax.axis('on')

    save_fig(save_dir=save_dir, title=title)


def plot_multi_channel_numpy_adjs(adjs, title='multi_channel_viz', save_dir=None):
    channel_nums = [adj.shape[0] for adj in adjs]
    x_max = int(np.max(channel_nums))
    y_max = len(adjs)
    figure = plt.figure(figsize=(x_max * 2, y_max * 2))
    cardinal_g = nx.from_numpy_array(np.asarray(adjs[0][0] > 0.5, dtype=np.float))
    pos = nx.spring_layout(cardinal_g)
    weight_ranges = [(-1e10, -0.8), (-0.8, 0.5), (0.5, 1e10)]
    cmap_list = [plt.cm.Blues_r, plt.cm.twilight_shifted, plt.cm.Reds]
    alphas = [0.9, 0.1, 0.9]
    widths = [0.3, 0.1, 0.3]

    for layer_i, adj in enumerate(adjs):
        assert isinstance(adj, np.ndarray)
        for channel_i in range(adj.shape[0]):
            ax = plt.subplot(y_max, x_max, layer_i * x_max + channel_i + 1)
            g = nx.from_numpy_array(adj[channel_i])
            assert isinstance(g, nx.Graph)
            g.remove_nodes_from(list(nx.isolates(g)))
            # pos = nx.spring_layout(g)
            plt.axis('off')
            # nodes
            nx.draw_networkx_nodes(g, pos, node_size=10)


            w = []
            group_num = len(weight_ranges)
            # for weight_range, cmap, alphas in zip(weight_ranges, cmap_list, alphas):
            e_groups = [[] for _ in range(group_num)]
            w_groups = [[] for _ in range(group_num)]
            for (u, v, d) in g.edges(data=True):
                weight = d['weight']
                w.append(weight)
                for group_i, weight_range in enumerate(weight_ranges):
                    if weight_range[0] < weight <= weight_range[1]:
                        e_groups[group_i].append((u, v))
                        w_groups[group_i].append(weight)
                        break
            for e_group, w_group, cmap, alpha, width in zip(e_groups, w_groups, cmap_list, alphas, widths):
                nx.draw_networkx_edges(g, pos, edgelist=e_group,
                                       edge_cmap=cmap, edge_color=w_group,
                                       alpha=alpha,
                                       width=width)
            ax.title.set_text(f'{layer_i}, {channel_i}, '
                              f'{"/".join([str(len(group)) for group in e_groups])}, \n'
                              f'{np.mean(w):.1e}, '
                              f'{np.std(w):.1e}')
    figure.suptitle(title)
    save_fig(save_dir=save_dir, title=title, dpi=300)


def plot_multi_channel_numpy_adjs_1b1(adjs, title='multi_channel_viz', save_dir=None, fig_dir='fig'):
    """plot the intermediate channels
    """
    suffix = '.' + title.split('.')[-1]
    file_name = title.rstrip(suffix)
    channel_nums = [adj.shape[0] for adj in adjs]
    x_max = int(np.max(channel_nums))
    y_max = len(adjs)
    a_min, a_max = -2.0, 2.0
    # adjs = np.clip(adjs, a_max=a_max, a_min=a_min)
    cardinal_g = nx.from_numpy_array(np.asarray(adjs[0][0] > 0.5, dtype=np.float))
    pos = nx.spring_layout(cardinal_g)

    for layer_i, adj in enumerate(adjs):
        assert isinstance(adj, np.ndarray)
        for channel_i in range(adj.shape[0]):
            g = nx.from_numpy_array(adj[channel_i])
            assert isinstance(g, nx.Graph)
            iso_nodes = list(nx.isolates(g))
            g.remove_nodes_from(iso_nodes)
            n = g.number_of_nodes()

            data = adj[channel_i, :n, :n]
            d_min = np.min(data)
            d_max = np.max(data)
            d_mean = np.mean(data)
            d_std = np.std(data)
            data = (data - d_mean) / d_std
            g = nx.from_numpy_array(data)
            plt.figure(figsize=(12 / 4, 12 / 4))
            k = 0.4
            colors1 = plt.cm.Blues_r(np.linspace(0, 1 - k / 2, 128))
            colors5 = plt.cm.twilight_shifted(np.linspace(0. + k, 1 - k, 128))
            colors2 = plt.cm.Reds(np.linspace(k / 2, 1, 128))

            # Combine them and build a new colormap
            colors = np.vstack((colors1, colors5, colors2))
            mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

            plt.pcolor(data, cmap=mymap, vmin=a_min, vmax=a_max)
            plt.axis(False)

            save_fig(save_dir=save_dir, title=file_name + f'_l{layer_i}_c{channel_i}_m' + suffix, dpi=300,
                     fig_dir=fig_dir)

            def draw_a_graph(g, g_suffix='g'):
                plt.figure(figsize=(12 / 4, 12 / 4))
                pos = nx.spring_layout(g)
                plt.axis('off')
                # nodes
                nx.draw_networkx_nodes(g, pos, node_size=10)
                w = []
                e = []
                for (u, v, d) in g.edges(data=True):
                    weight = d['weight']
                    e.append((u, v))
                    w.append(weight)
                nx.draw_networkx_edges(g, pos, edgelist=e,
                                       edge_cmap=mymap, edge_color=w, edge_vmin=a_min,
                                       edge_vmax=a_max,
                                       width=0.9,
                                       alpha=0.5)
                save_fig(save_dir=save_dir, title=file_name + f'_l{layer_i}_c{channel_i}_{g_suffix}' + suffix, dpi=300,
                         fig_dir=fig_dir)

            draw_a_graph(g, g_suffix='g')
            comp_g = nx.from_numpy_array(- data)
            draw_a_graph(comp_g, g_suffix='c')


def save_fig(save_dir=None, title='fig', dpi=300, fig_dir='fig'):
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    if save_dir is None:
        plt.show()
    else:
        fig_dir = os.path.join(save_dir, fig_dir)
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        plt.savefig(os.path.join(fig_dir, title),
                    bbox_inches='tight',
                    dpi=dpi,
                    transparent=True)
        plt.close()
    return


def plot_curve(ax=None, data=None, title='energy', draw_std=False):
    if ax is None:
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
    infos = np.asarray(data)
    if draw_std:
        infos = pd.DataFrame(infos, columns=['', title, 'std'])
        plt.plot(infos[''], infos[title], 'b')
        plt.fill_between(infos[''], infos[title] - infos['std'], infos[title] + infos['std'], color='b', alpha=0.2)
        infos.plot(0, [1], ax=ax, grid=True)
        ax.axhline(y=0, color='k', linewidth=1.2)
        ax.axvline(x=0, color='k', linewidth=1.2)
        ax.axvline(x=1, color='k', linewidth=1.2)
    else:
        infos = pd.DataFrame(infos, columns=['', title])
        infos.plot(0, [1], ax=ax, grid=True)


def plot_graphs_list(graphs, energy=None, node_energy_list=None, title='title', max_num=16, save_dir=None):
    batch_size = len(graphs)
    max_num = 16
    img_c = int(np.sqrt(max_num)) 
    figure = plt.figure()

    for i in range(max_num):
        #idx = i * (batch_size // max_num)
        idx=i
        if not isinstance(graphs[idx], nx.Graph):
            G = graphs[idx].g.copy()
        else:
            G = graphs[idx].copy()
        assert isinstance(G, nx.Graph)
        G.remove_nodes_from(list(nx.isolates(G)))
        e = G.number_of_edges()
        v = G.number_of_nodes()
        l = nx.number_of_selfloops(G)
        ax = plt.subplot(img_c, img_c, i + 1)
        title_str = f'e={e - l}, n={v}'
        if energy is not None:
            title_str += f'\n en={energy[idx]:.1e}'
        if node_energy_list is not None:
            node_energy = node_energy_list[idx]
            title_str += f'\n {np.std(node_energy):.1e}'
            nx.draw(G, with_labels=False, node_color=node_energy, cmap=cm.jet, **options)
        else:
            # print(nx.get_node_attributes(G, 'feature'))
            pos = nx.spring_layout(G)
            nx.draw(G, pos, with_labels=False, **options)
        ax.title.set_text(title_str)
    figure.suptitle(title)
    save_fig(save_dir=save_dir, title=title)


def plot_inter_graphs(graphs,flags,title,save_dir,nr_to_analyze=0):
    torch.set_printoptions(profile="full")
    figure = plt.figure()
    max_num=len(graphs)
    img_c = int(np.sqrt(max_num))
    for i, sigmalevel_adjs in enumerate(graphs):
        nodes=flags.sum(-1)
        G = sigmalevel_adjs[nr_to_analyze].copy()
        G.remove_nodes_from(list(nx.isolates(G)))
        e = G.number_of_edges()
        v = G.number_of_nodes()
        l = nx.number_of_selfloops(G)
        ax = plt.subplot(img_c+1, img_c+1, i+1)
        title_str = f'fl={"nrofnodes"},no={i}'
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=False, **options)
        ax.title.set_text(title_str)
        plt.show()
        
    title=f'{title}-{nr_to_analyze}.pdf'
    figure.suptitle(title)
    save_fig(save_dir=save_dir, title=title)

def plot_inter_graphs_jup(graphs,flags,title,save_dir,nr_to_analyze=0):
    maxnodes=0
    graphs_proces=[]
    for g in graphs:
        graphs_proces.append(nx.from_numpy_matrix(g[0]))
    torch.set_printoptions(profile="full")
    figure = plt.figure(figsize=(25, 25))
    max_num=len(graphs)
    img_c = int(np.sqrt(max_num))
    pos=nx.spring_layout(graphs_proces[-1])
    for i, sigmalevel_adjs in enumerate(graphs_proces):
        nodes=flags.sum(-1)
        G = sigmalevel_adjs.copy()
        G.remove_nodes_from(list(nx.isolates(G)))
        e = G.number_of_edges()
        v = G.number_of_nodes()
        l = nx.number_of_selfloops(G)
        ax = plt.subplot(img_c+1, img_c+1, i+1)
        title_str = f'fl={"nrofnodes"},no={i}'
        nx.draw(G, pos, with_labels=False, **options)
        ax.title.set_text(title_str)
        plt.show()
    title=f'{title}-{nr_to_analyze}.pdf'
    figure.suptitle(title)
    save_fig(save_dir=save_dir, title=title)


def plot_inter_graphs_j(graphs,flags,title,save_dir,nr_to_analyze=0):
    torch.set_printoptions(profile="full")
    figure = plt.figure(figsize=(15, 15))
    max_num=len(graphs)
    img_c = int(np.sqrt(max_num))
    pos=nx.spring_layout(graphs[-1])
    for i, sigmalevel_adjs in enumerate(graphs):
        nodes=flags.sum(-1)
        G = sigmalevel_adjs[nr_to_analyze].copy()
        G.remove_nodes_from(list(nx.isolates(G)))
        e = G.number_of_edges()
        v = G.number_of_nodes()
        l = nx.number_of_selfloops(G)
        ax = plt.subplot(img_c+1, img_c+1, i+1)
        title_str = f'fl={"nrofnodes"},no={i}'
            # print(nx.get_node_attributes(G, 'feature'))
        nx.draw(G, pos, with_labels=False, **options)
        ax.title.set_text(title_str)
        plt.show()
    title=f'{title}-{nr_to_analyze}.pdf'
    figure.suptitle(title)
    save_fig(save_dir=save_dir, title=title)


# Same as plot intergraphs but expects a list as an input to nrtoanalyze
def plot_inter_graphs_list(graphs,flags,title,save_dir,nr_to_analyze=0):
    for i in nr_to_analyze:
        plot_inter_graphs(graphs,flags,title,save_dir,nr_to_analyze=i)
    return


def plot_graphs_adj(adjs, energy=None, node_num=None, title='title', max_num=16, save_dir=None):
    if isinstance(adjs, torch.Tensor):
        adjs = adjs.cpu().numpy()
    with_labels = (adjs.shape[-1] < 10)
    batch_size = adjs.shape[0]
    max_num = min(batch_size, max_num)
    img_c = np.ceil(np.sqrt(int(max_num)))
    figure = plt.figure()
    for i in range(max_num):
        idx = i * (adjs.shape[0] // max_num)
        adj = adjs[idx, :, :]
        G = nx.from_numpy_matrix(adj)
        assert isinstance(G, nx.Graph)
        G.remove_edges_from(list(nx.selfloop_edges(G)))
        G.remove_nodes_from(list(nx.isolates(G)))
        e = G.number_of_edges()
        v = G.number_of_nodes()
        l = nx.number_of_selfloops(G)
        ax = plt.subplot(img_c, img_c, i + 1)
        title_str = f'e={e - l}, n={v}'
        if energy is not None:
            title_str += f'\n en={energy[idx]:.1e}'
        ax.title.set_text(title_str)
        nx.draw(G, with_labels=with_labels, **options)
    figure.suptitle(title)
    save_fig(save_dir=save_dir, title=title)