import gc
import sys

import torch
    
def torchlisttensors(*roots):
    # https://github.com/szagoruyko/pytorchviz/blob/0adcd83af8aa7ab36d6afd139cabbd9df598edb7/torchviz/dot.py#L146

    seen = set()
    nodes = {}

    def found_tensor_hook(tensor, name):
        if id(tensor) not in nodes:
            nodes[id(tensor)] = name
    
    def get_var_name(var, name = '', from_var = None):
        if not name:
            #name = param_map[id(var)] if id(var) in param_map else ''
            if hasattr(var, '__name__'):
                name = var.__name__
            else:
                name = '__' + str(id(var)) + '__'
        return 'shape=({})\tname={}'.format(','.join(map(str, var.shape)), name)

    def add_nodes(fn, from_var = None):
        assert not torch.is_tensor(fn)
        if fn in seen:
            return
        seen.add(fn)

        fmtvarname = lambda varname, saved = False: varname + '__{}_{}'.format(type(fn).__name__, from_var) + ('_SAVED' * saved)

        SAVED_PREFIX = "_saved_"
        for attr in dir(fn):
            if not attr.startswith(SAVED_PREFIX):
                continue
            val = getattr(fn, attr)
            seen.add(val)
            attr = attr[len(SAVED_PREFIX):]
            if torch.is_tensor(val):
                varname = get_var_name(val, attr, from_var = id(fn) )
                found_tensor_hook(val, fmtvarname(varname, saved = True))
            if isinstance(val, tuple):
                for i, t in enumerate(val):
                    if torch.is_tensor(t):
                        varname = get_var_name(t, attr + '[{}]'.format(i), from_var = id(fn))
                        found_tensor_hook(t, fmtvarname(varname, saved = True))

        if hasattr(fn, 'variable'):
            var = fn.variable
            seen.add(var)
            varname = get_var_name(var, from_var = id(fn))
            found_tensor_hook(var, fmtvarname(varname))

        if hasattr(fn, 'next_functions'):
            for u in fn.next_functions:
                if u[0] is not None:
                    add_nodes(u[0], from_var = id(fn))

        if hasattr(fn, 'saved_tensors'):
            for t in fn.saved_tensors:
                seen.add(t)
                varname = get_var_name(t, from_var = id(fn))
                found_tensor_hook(t, fmtvarname(varname))

    def add_base_tensor(var):
        if var in seen:
            return
        seen.add(var)
        found_tensor_hook(var, get_var_name(var))
        if var.grad_fn:
            add_nodes(var.grad_fn, from_var = id(var))
        if var._is_view():
            add_base_tensor(var._base, from_var = id(var))

    seeds = roots[:]
    for objs in [gc.get_objects(), sys.getobjects() if hasattr(sys, 'getobjects') else []]:
        for obj in objs:
            try:
                if torch.is_tensor(obj):
                    tensor = obj
                elif hasattr(obj, 'data') and torch.is_tensor(obj.data):
                    tensor = obj.data
                else:
                    continue
                if hasattr(tensor, '__name__'):
                    found_tensor_hook(tensor, get_var_name(tensor))
                elif tensor not in seeds:
                    seeds.append(tensor)
            except Exception as e:
                pass
    
    for tensor in seeds:
        add_base_tensor(tensor)

    return nodes

def assign_names(model, name):
    for __name__, m in model.named_modules(prefix = name):
        m.__name__ = __name__
    for __name__, p in model.named_parameters(prefix = name):
        p.__name__ = __name__
    for __name__, b in model.named_buffers(prefix = name):
        b.__name__ = __name__
    # TODO: also go over tensor attributes, stored in modules
    return model
    
def assign_names_output_hook(module, input, output):
    __name__ = (module.__name__ if hasattr(module, '__name__') else type(module).__name__) + '_output'
    if torch.is_tensor(output) and (not hasattr(output, '__name__')):
        output.__name__ = __name__
    if isinstance(output, (list, tuple)):
        for i, output_i in output:
            if torch.is_tensor(output_i) and (not hasattr(output_i, '__name__')):
                output_i.__name__ = __name__ + str(i)

if __name__ == '__main__':
    model = torch.nn.Sequential(torch.nn.Linear(20, 20), torch.nn.Linear(20, 20), torch.nn.Linear(20, 20))
    model = assign_names(model, 'model')
    model.apply(lambda module: module.register_forward_hook(assign_names_output_hook))

    x = torch.zeros(4, 20)
    x.__name__ = 'x'

    loss = model(x).sum()
    loss.__name__ = 'loss'

    z = torch.zeros(65, 35)
    z.__name__ = 'z'

    tensors = torchlisttensors(loss)
    for i, t in tensors.items():
        print(i, '\t', t)
