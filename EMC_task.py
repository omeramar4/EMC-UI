
import pandas as pd
import networkx as nx
import numpy as np
import seaborn as sns
from bokeh.models import Range1d, MultiLine, Circle, HoverTool, Button, TapTool, ColumnDataSource
from bokeh.models.graphs import from_networkx, NodesAndLinkedEdges, EdgesAndLinkedNodes
from bokeh.plotting import figure, curdoc
from bokeh.layouts import row, column


# Arrange data by running on every pair of users (O(n^2))
def create_average_df_1(data_frame, num_of_users, columns):
    averages = pd.DataFrame()
    for i in range(1, num_of_users):
        user_i_data = data_frame[data_frame['UserID'] == i]
        for j in range(i + 1, num_of_users):
            user_j_data = (data_frame[data_frame['UserID'] == j])
            df1 = user_i_data[user_i_data['ItemID'].isin(user_j_data['ItemID'].values)]
            if df1.shape[0] == 0:
                continue
            df2 = user_j_data[user_j_data['ItemID'].isin(user_i_data['ItemID'].values)]
            items = df1["ItemID"].tolist()
            avg = np.mean(abs(np.array(df1["Rating"]) - np.array(df2["Rating"])))
            averages = averages.append({"Node1": i, "Node2": j, "Items": items, "Count": len(items), "Average": avg},
                                       ignore_index=True)
        print(i)
    averages["Node1"] = averages["Node1"].astype(int)
    averages["Node2"] = averages["Node2"].astype(int)
    averages["Count"] = averages["Count"].astype(int)
    averages = averages[columns]
    return averages


# Arrange data by running on users and items
def create_average_df_2(data_frame, num_of_users):
    averages = pd.DataFrame()
    for i in range(1, num_of_users):
        temp_data1 = data_frame[data_frame['UserID'] == i]
        items = temp_data1["ItemID"].tolist()
        td = data_frame[data_frame["UserID"] > i]
        td = td[td['ItemID'].isin(temp_data1['ItemID'].values)]
        for k in range(len(items)):
            df = td[td["ItemID"] == items[k]]
            df["Rating"] = abs(df["Rating"] - items[k])
            td[td["ItemID"] == items[k]] = df
        avg = (td.groupby(['UserID'])['Rating'].mean()).tolist()
        node2 = set(td["UserID"].tolist())
        temp_list = [[i, y, z] for y, z in zip(node2, avg)]
        averages = averages.append(pd.DataFrame(temp_list, columns=cols))
        print(i)
    return averages


# Create sizes and colors dictionaries by degree
def set_size_color(g):
    degrees = [k[1] for k in g.degree()]  # List of node degrees by order
    max_deg, min_deg = max(degrees), min(degrees)  # Keep max and min degrees for node coloring purpose
    node_size = {k: size_factor*v + 2 for k, v in g.degree()}  # Assign node size by degree

    # Assign node color by degree
    node_color = {k: v[1] for k, v in enumerate(g.degree())}
    palette = sns.color_palette('bright', color_range + 1)
    pal_hex_lst = list(palette.as_hex())
    pal_hex_lst.sort()
    for k, v in node_color.items():
        node_color[k] = round(((node_color[k] - min_deg) * color_range) / (max_deg - min_deg))
        node_color[k] = pal_hex_lst[node_color[k]]
    for n in range(len(users), 0, -1):
        node_color[n] = node_color.pop(n - 1)
    return node_size, node_color


# Create graph with nodes and edges and add properties
def create_graph(g, g2, plt, is_isolated, edge):
    global users_with_edges
    if is_isolated == 0:
        g.add_nodes_from(users)  # Add every user as a node (vertex)
        g.add_edges_from(edge)
        node_size, node_color = set_size_color(g)
    else:
        g.add_nodes_from(users)
        g.add_edges_from(edge)
        isolates = list(nx.isolates(g2))  # Find isolated nodes (no edge)
        users_with_edges = [x for x in users if x not in isolates]
        g.remove_nodes_from(isolates)
        node_size, node_color = set_size_color(g2)

    # Add size and color attributes to the graph
    nx.set_node_attributes(g, node_size, 'node_size')
    nx.set_node_attributes(g, node_color, 'node_color')
    source = ColumnDataSource(pd.DataFrame.from_dict({k: v for k, v in g.nodes(data=True)}, orient='index'))

    # Create a graph from the NetworkX input using nx.spring_layout
    graph = from_networkx(g, nx.spring_layout, scale=50000, center=(0, 0))

    graph.node_renderer.data_source = source
    graph.node_renderer.glyph = Circle(size='node_size', fill_color='node_color')  # Add node sizes and colors
    graph.node_renderer.selection_glyph = Circle(fill_color="#00FF00")  # Change node color on selection
    graph.edge_renderer.glyph = MultiLine(line_color="#cccccc", line_alpha=1, line_width=2)  # Edge colors
    graph.edge_renderer.selection_glyph = MultiLine(line_color="#222222", line_width=5)  # Edges size & color on select

    # green hover for both nodes and edges
    graph.node_renderer.hover_glyph = Circle(fill_color='#abdda4')
    graph.edge_renderer.hover_glyph = MultiLine(line_color='#FF0000')

    plt.renderers.append(graph)

    return graph


# Connect nodes by thresholds
def link_edges():
    global pass_df, edges, item_label
    pass_df = average_df[average_df["Average"] <= avg_th]
    pass_df = pass_df[pass_df["Count"] >= num_of_items]
    edges = list(zip(pass_df["Node1"].tolist(), pass_df["Node2"].tolist()))
    item_label = pass_df["Items"].tolist()  # Keep items lists for edge hover ability


# Open and sort data in pandas data frame
data = pd.read_csv('ml.data', sep='\t', skiprows=1, names=['UserID', 'ItemID', 'Rating', 'TimeStamp'])
data.sort_values(['UserID', 'ItemID'], inplace=True)

users = set(data['UserID'])     # list of users
N = max(users) + 1      # for the loop
cols = ["Node1", "Node2", "Items", "Count", "Average"]      # Columns of averages data frame

choice = 3  # 1 - use create_average_df_1 function, 2 - use create_average_df_2 function, 3 - read saved csv file
if choice == 1:
    average_df = create_average_df_1(data, N, cols)
    average_df.to_pickle("averages2.csv")
elif choice == 2:
    average_df = create_average_df_2(data, N)
    average_df.to_pickle("averages2.csv")
else:
    average_df = pd.read_pickle("averages.csv")

# Global variables
# --------------------------------------------------------------------------
flag_isolates = 0    # show/hide isolated nodes
flag_inspection = 0  # 0 - edges, 1 - nodes
avg_th = 0           # Default average difference threshold
num_of_items = 4     # Default number of common items threshold
ax_range = 45000     # Starting axes range
plot_width = 1500
plot_height = 610
size_factor = 5
color_range = 20
doc = curdoc()      # The page of the plot (network)
users_with_edges = users
graph_list = []
G_list = []
plot_list = []
G = nx.Graph()      # Create graph object with networkx
plot = figure(x_range=Range1d(-ax_range, ax_range), y_range=Range1d(-ax_range, ax_range), plot_width=plot_width,
              plot_height=plot_height, tools="reset,pan,wheel_zoom,lasso_select")
# --------------------------------------------------------------------------

link_edges()  # Filter by average difference and number of shared items thresholds

new_graph = create_graph(G, G, plot, 0, edges)      # Create default graph

# Append to lists
graph_list.append(new_graph)
G_list.append(G)
plot_list.append(plot)

# Buttons
# --------------------------------------------------------------------------
isolate_btn = Button(label="Show/Hide isolated nodes", button_type="primary")
inspection_btn = Button(label="Change Hover Method", button_type="primary")
avg_th_up_btn = Button(label="Increase average threshold by 0.1", button_type="primary")
avg_th_down_btn = Button(label="Decrease average threshold by 0.1", button_type="primary")
noi_th_up_btn = Button(label="Increase number of items threshold by 1", button_type="primary")
noi_th_down_btn = Button(label="Decrease number of items threshold by 1", button_type="primary")
# --------------------------------------------------------------------------

# Main layout - starting plot
layout = row(plot)


# Buttons functions
# --------------------------------------------------------------------------
# Show/Hide isolated nodes
def change_isolate():
    global flag_isolates, edges
    g_new = nx.Graph()
    plt_new = figure(x_range=Range1d(-ax_range, ax_range), y_range=Range1d(-ax_range, ax_range), plot_width=plot_width,
                     plot_height=plot_height, tools="reset,pan,wheel_zoom,lasso_select")
    if flag_isolates == 0:
        flag_isolates = 1
        graph_new = create_graph(g_new, G_list[-1], plt_new, 1, edges)
    else:
        flag_isolates = 0
        graph_new = create_graph(g_new, g_new, plt_new, 0, edges)
    del graph_list[0]
    del G_list[0]
    G_list.append(g_new)
    graph_list.append(graph_new)
    plot_list.append(plt_new)
    change_inspection()
    layout.children[0] = plt_new


# Toggle nodes and edges inspection
def change_inspection():
    global flag_inspection
    graph_list[-1].selection_policy = NodesAndLinkedEdges()
    if flag_inspection == 0:
        flag_inspection = 1
        hover_edge = HoverTool(tooltips=[("Items", "@items")])
        plot_list[-1].add_tools(hover_edge, TapTool())
        graph_list[-1].inspection_policy = EdgesAndLinkedNodes()
        graph_list[-1].edge_renderer.data_source.data['items'] = [','.join(map(str, k)) for k in item_label]
        plot_list[-1].renderers.append(graph_list[-1])
    else:
        flag_inspection = 0
        hover_node = HoverTool(tooltips=[("User ID", "@id")])
        plot_list[-1].add_tools(hover_node, TapTool())
        graph_list[-1].inspection_policy = NodesAndLinkedEdges()
        if flag_isolates == 0:
            graph_list[-1].node_renderer.data_source.data['id'] = [str(k) for k in users]
        else:
            graph_list[-1].node_renderer.data_source.data['id'] = [str(k) for k in users_with_edges]
        plot_list[-1].renderers.append(graph_list[-1])


# Update plot after average or number of items threshold changed
def threshold_update():
    global item_label, pass_df, edges, avg_th, num_of_items, flag_isolates
    flag_isolates = 0
    link_edges()
    g_new = nx.Graph()
    plt_new = figure(x_range=Range1d(-ax_range, ax_range), y_range=Range1d(-ax_range, ax_range), plot_width=plot_width,
                     plot_height=plot_height, tools="reset,pan,wheel_zoom,lasso_select")
    graph_new = create_graph(g_new, g_new, plt_new, 0, edges)
    del graph_list[0]
    del G_list[0]
    graph_list.append(graph_new)
    G_list.append(g_new)
    plot_list.append(plt_new)
    change_inspection()
    layout.children[0] = plt_new


# Increase average difference threshold by 0.1
def avg_threshold_up():
    global avg_th
    if avg_th == 1:
        return
    avg_th += 0.1
    threshold_update()


# Decrease average difference threshold by 0.1
def avg_threshold_down():
    global avg_th
    if avg_th == 0:
        return
    avg_th -= 0.1
    threshold_update()


# Increase number of items threshold by 1
def noi_threshold_up():
    global num_of_items
    if num_of_items == 10:
        return
    num_of_items += 1
    threshold_update()


# Decrease number of items threshold by 1
def noi_threshold_down():
    global num_of_items
    if num_of_items == 1:
        return
    num_of_items -= 1
    threshold_update()
# --------------------------------------------------------------------------


# Button click events
# --------------------------------------------------------------------------
isolate_btn.on_click(change_isolate)
inspection_btn.on_click(change_inspection)
avg_th_up_btn.on_click(avg_threshold_up)
avg_th_down_btn.on_click(avg_threshold_down)
noi_th_up_btn.on_click(noi_threshold_up)
noi_th_down_btn.on_click(noi_threshold_down)
# --------------------------------------------------------------------------

change_inspection()  # Default inspection policy

# The web page
doc.title = 'dellEMC-Dashboard'
doc.add_root(column(layout, row(isolate_btn, inspection_btn, column(avg_th_up_btn, avg_th_down_btn),
                                column(noi_th_up_btn, noi_th_down_btn))))
