import ast
import traceback
import json


def value_to_dict(val) -> dict:
    """Get dict referring to either Value instance or a constant within DAG"""
    if isinstance(val, Value):
        return {
            'type': 'value',
            'id': val.dag_idx
        }
    else:
        return {
            'type': 'constant',
            'value': val
        }


class Dag:
    def __init__(self, name=None):
        self.name = name
        self.nodes = []
        self.vals = []

    def __getattr__(self, item):
        """Create new node with name as specified attribute and return it

        Allows creation of multiple nodes with the same name (different indexes will be used)
        """
        idx = len(self.nodes)
        node = Node(dag=self, idx=idx, name=item)
        self.nodes.append(node)
        return node

    def __call__(self) -> dict:
        """Prepare DAG to JSON serialization, return dict"""
        ans = {
            'nodes': [
                {
                    'name': n.name,
                    'args': [
                        value_to_dict(i)
                        for i in n.args
                    ],
                    'kwargs': {
                        k: value_to_dict(v)
                        for k, v in n.kwargs.items()
                    }
                }
                for n in self.nodes
            ],
            'values': [
                {
                    'src_node_idx': v.src_node.idx,
                    'src_output_idx': v.src_output_idx,
                    'name': v.name,
                }
                for v in self.vals
            ]
        }
        return ans

    def __getitem__(self, item):
        """Create new Value within DAG

        By default created Value not connected to any Node, but will be serialized anyway"""
        val = Value(len(self.vals), name=item)
        self.vals.append(val)
        return val


class Node:
    """Node of a DAG. Store info about args/kwargs, results(outputs)"""
    def __init__(self, dag, idx, name=None):
        self.dag = dag
        self.idx = idx
        self.name = name
        self.args = ()
        self.kwargs = {}
        self.output = None

    def __call__(self, *args, **kwargs):
        """Account args, kwargs. Return and account as many answers as needed for unpacking

        Support returning multiple values (number of values defined via introspection trick)
        I.e. for the line `b, c = node(a, param=val)`
            node will save input argument `a` and keyword argument `val` for key `param`
            node will create 2 new Values within current DAG, name them `b` and `c` and return them"""
        self.args = args
        self.kwargs = kwargs

        # Generate output
        try:
            raise DummyException()
        except DummyException:
            code_line = traceback.extract_stack()[0].line
            ast_obj = ast.parse(code_line)
            assert isinstance(ast_obj, ast.Module), "Not ast.Module. Couldn't proceed"
            body = ast_obj.body
            assert (len(body) == 1), "Not just single assignment in the line of code with call of the node"
            assign = body[0]
            if isinstance(assign, ast.AnnAssign):
                target = assign.target
                # TODO: process annotations
                val = self.dag[None]
                val.src_node = self
                val.src_output_idx = 0
                if isinstance(target, ast.Name):
                    val.name = target.id
                self.output = val
                return val
            elif isinstance(assign, ast.Expr):
                # that assumed to be a call like `node(something)` without results
                # TODO: prove it's a correct function call
                return None
            assert isinstance(assign, ast.Assign), "Not an assignment on the call of node"
            targets = assign.targets
            assert (len(targets) == 1), "More than one target for an assignment"
            target = targets[0]
            if isinstance(target, (ast.Tuple, ast.List)):
                vals = []
                for i, e in enumerate(target.elts):
                    val = self.dag[None]
                    val.src_node = self
                    val.src_output_idx = i
                    if isinstance(e, ast.Name):
                        val.name = e.id
                    vals.append(val)
                self.output = vals
                return vals
            else:
                val = self.dag[None]
                val.src_node = self
                val.src_output_idx = 0
                if isinstance(target, ast.Name):
                    val.name = target.id
                self.output = val
                return val

    def __repr__(self):
        node_name = f'{self.name if self.name else self.idx}'
        node_input = f'{", ".join(str(i) for i in self.args)}'
        node_output = f' -> {self.output}' if self.output else ''
        ans = f'<Node {node_name}({node_input}){node_output}>'
        return ans


class Value:
    """Used as args for nodes and as results of a node call. Represent data between nodes"""
    def __init__(self, dag_idx, src_node=None, src_output_idx=0, name=None):
        self.dag_idx = dag_idx
        self.src_node = src_node
        self.src_output_idx = src_output_idx
        self.name = name

    def __repr__(self):
        val_name = f': {self.name}' if self.name else ''
        ans = f'<Value[{self.dag_idx}]{val_name}>'
        return ans


class DummyException(Exception):
    """Used for introspection trick with returning values by a node"""
    pass


if __name__ == '__main__':
    d = Dag('Example DAG')

    data = d.get_data()
    X_train, X_test, y_train, y_test = d.split_data(data)
    preprocessor, X_train = d.create_preprocess_data(X_train)
    model, metrics_train = d.create_evaluate_model(X_train, y_train)
    X_test = d.preprocess_data(preprocessor, X_test)
    metrics_test = d.evaluate_model(model, X_test, y_test)
    d.visualize_metrics(metrics_train, metrics_test)

    dag_json = d()
    with open('example_dag.json', 'w') as f:
        json.dump(dag_json, f, indent=2)
    print(json.dumps(dag_json, indent=2))
