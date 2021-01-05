"""Poorly structured code, works not for any DAG. Just for some terminal visualization"""
from dag_gen import Value, Dag


def pretty_print_dag(dag):
    nodes_to_print = {i: n for i, n in enumerate(dag.nodes)}

    # organize nodes into columns and rows
    node_columns = []  # columns of prepared nodes
    prev_output_pos = {}
    while len(nodes_to_print) > 0:  # while have some nodes to print
        current_column_nodes = []  # nodes with all parents printed
        for n in nodes_to_print.values():  # fill `current_step_nodes`
            parent_idxs = get_parent_idxs(n)
            if all(idx not in nodes_to_print for idx in parent_idxs):
                current_column_nodes.append(n)
        for n in current_column_nodes:
            del nodes_to_print[n.idx]

        # sort
        for val, (x, y) in prev_output_pos.items():
            prev_output_pos[val] = x + 1, y
        if len(node_columns) > 0:
            for n in node_columns[-1]['nodes']:
                for val, (x, y) in n['output_pos'].items():
                    prev_output_pos[val] = x, y

        sorted_column_nodes = sorted(current_column_nodes,
                                     key=lambda target_node: node_vertical_order_key(target_node, prev_output_pos))

        prepared_nodes = [prepare_node(n) for n in sorted_column_nodes]
        outputs_pos = {}
        for i, n in enumerate(prepared_nodes):
            for val in n['output_pos'].keys():
                outputs_pos[val] = 0, i

        node_columns.append({
            'nodes': prepared_nodes,
            'output_pos': outputs_pos
        })
    print(f'<<{dag.name}>>')
    print_prepared(node_columns)


def get_parent_idxs(node):
    parent_vals = []
    for v in node.args:
        if isinstance(v, Value):
            parent_vals.append(v)
    for v in node.kwargs.values():
        if isinstance(v, Value):
            parent_vals.append(v)

    ans = []
    for v in parent_vals:
        if v.src_node:
            ans.append(v.src_node.idx)
    return ans


def node_vertical_order_key(node, vals_coordinates):
    """

    1. Max horizontal distance to source val
    2. Min horizontal distance to dependent node # TODO: implement
    """
    farthest_source = 0
    for val in node.args:
        hor_dst, ver_pos = vals_coordinates.get(val, (0, 0))
        farthest_source = max(farthest_source, hor_dst)
    return farthest_source


def prepare_node(node):
    output_pos = {}

    node_name = str(node.name) if node.name else ''
    inputs = [*node.args, *node.kwargs.values()]
    n_inputs = len(inputs)
    n_outputs = 0
    if isinstance(node.output, Value):
        n_outputs = 1
    elif isinstance(node.output, list):
        n_outputs = len(node.output)
    node_height = max(1, n_inputs, n_outputs)
    node_width = len(node_name)

    node_text = ['']*(node_height+2)
    node_text[0] = '┌' + '─' * node_width + '┐'
    for i in range(1, node_height+1):
        if n_inputs > 0:
            node_text[i] += '┤'
            n_inputs -= 1
        else:
            node_text[i] += '│'

        node_text[i] += f'{node_name:^{node_width}}'
        node_name = ''
        if n_outputs > 0:
            output_val = node.output[-n_outputs] if isinstance(node.output, list) else node.output
            output_name = output_val.name
            node_text[i] += f'├{output_name}─'
            output_pos[output_val] = i, len(node_text[i])
            n_outputs -= 1
        else:
            node_text[i] += '│'
    node_text[-1] = '└' + '─' * node_width + '┘'

    prepared_node = {
        'node_text': node_text,
        'output_pos': output_pos,
        'inputs': inputs
    }
    return prepared_node


def print_prepared(node_columns):
    absolute_output_pos = {}
    lines = []
    current_x = 0
    for column in node_columns:
        current_y = 0
        for n in column['nodes']:
            for i, inp_val in enumerate(n['inputs']):
                if current_y+i+1 < absolute_output_pos[inp_val][1]:
                    current_y += absolute_output_pos[inp_val][1] - (current_y+i+1)
                while obstacles_here(lines, absolute_output_pos[inp_val][0]+1, current_x-1, current_y+i+1):
                    current_y += 1
                # print(absolute_output_pos[inp_val], (current_x-1, current_y+i+1))
            add_node_text_to_lines(lines, n['node_text'], current_x, current_y)
            for i, inp_val in enumerate(n['inputs']):
                connect_line(lines, absolute_output_pos[inp_val], (current_x-1, current_y+i+1))
            for val, (y_pos, x_pos) in n['output_pos'].items():
                absolute_output_pos[val] = 0, y_pos+current_y
            current_y += len(n['node_text'])
        current_x = len(lines[-1])
        for n in column['nodes'][::-1]:
            for val, (y_pos, x_pos) in reversed(n['output_pos'].items()):
                current_y = absolute_output_pos[val][1] + len(n['node_text']) - y_pos  # pos of the current node bottom
                current_x = max(current_x, len(lines[absolute_output_pos[val][1]]))
                # if len(lines[y_pos+current_y - len(n['node_text'])]) < last_val_x:
                #     print('!!!')
                end_of_max_box_offset = 0
                if len(lines[y_pos+current_y - len(n['node_text'])]) < current_x:
                    end_of_max_box_offset = current_x - len(lines[y_pos+current_y - len(n['node_text'])]) + 1
                # n_processed_vals = len([v for v in absolute_output_pos.values() if v[0] != 0])
                n_processed_vals = 0
                lines[y_pos + current_y - len(n['node_text'])] += '-'*(n_processed_vals + end_of_max_box_offset)
                lines[y_pos + current_y - len(n['node_text'])] = lines[y_pos+current_y - len(n['node_text'])][:-1] + '*'
                last_val_x = len(lines[y_pos+current_y - len(n['node_text'])])
                absolute_output_pos[val] = last_val_x-1, absolute_output_pos[val][1]
                current_x += 1
            # current_y -= len(n['node_text'])
        current_x = max(len(L) for L in lines)
    for L in lines:
        print(L)


def add_node_text_to_lines(lines, n_text, x, y):
    if len(lines) < y+len(n_text):
        lines.extend(['']*(y+len(n_text) - len(lines)))
    for i, L in enumerate(n_text):
        y_abs = i+y
        if len(lines[y_abs]) < x:
            lines[y_abs] += ' '*(x - len(lines[y_abs]))
        lines[y_abs] += L


def connect_line(lines, pos_from, pos_to):
    x_f, y_f = pos_from
    # print('From', lines[y_f][x_f])
    x_t, y_t = pos_to
    assert(x_f <= x_t), "Invalid x"
    assert(y_f <= y_t), "Invalid y"

    for i in range(y_t - y_f):
        if x_f >= len(lines[y_f+i]):
            lines[y_f + i] += ' ' * (x_f - len(lines[y_f+i]) + 1)
        s = lines[y_f+i][x_f]
        if s == ' ':
            replace_s(lines, x_f, y_f+i, '│')
        elif s == '─':
            # Long dash
            replace_s(lines, x_f, y_f+i, '┬')
        elif s == '-':
            # Short dash
            replace_s(lines, x_f, y_f+i, '┼')
        elif s == '└':
            replace_s(lines, x_f, y_f+i, '├')
        elif s == '*':
            replace_s(lines, x_f, y_f+i, '┐')

        if i == y_t - y_f - 1:
            replace_s(lines, x_f, y_f+i+1, '└')
            # print(x_f, y_f+i+1, "Change source")

    for i in range(x_t - x_f+1):
        s = lines[y_t][x_f+i]
        if s == ' ':
            replace_s(lines, x_f+i, y_t, '-')
        elif s == '│':
            # print('Replace')
            replace_s(lines, x_f+i, y_t, '┼')
        elif s == '*':
            replace_s(lines, x_f+i, y_t, '─')


def replace_s(lines, x, y, s):
    lines[y] = lines[y][:x] + s + lines[y][x+1:]


def obstacles_here(lines, x_from, x_to, y):
    if len(lines) <= y:
        return False
    for y_i in range(y, len(lines)):
        for i in range(x_from, x_to):
            if len(lines[y_i]) <= i:
                continue
            s = lines[y_i][i]
            if s != ' ':
                return True
    return False


if __name__ == '__main__':
    d = Dag('Example DAG')

    data = d.get_data()
    X_train, X_test, y_train, y_test = d.split_data(data)
    preprocessor, X_train = d.create_preprocess_data(X_train)
    model, metrics_train = d.create_evaluate_model(X_train, y_train)
    X_test = d.preprocess_data(preprocessor, X_test)
    metrics_test = d.evaluate_model(model, X_test, y_test)
    d.visualize_metrics(metrics_train, metrics_test)

    pretty_print_dag(d)
