def summary(input_size, model, device):
    import torch
    from collections import OrderedDict
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split('.')[-1].split("'")[0]
            module_idx = len(summary)

            m_key = '%s-%i' % (class_name, module_idx+1)
            summary[m_key] = OrderedDict()
            summary[m_key]['input_shape'] = list(input[0].size())
            summary[m_key]['input_shape'][0] = -1
            summary[m_key]['output_shape'] = list(output.size())
            summary[m_key]['output_shape'][0] = -1

            params = torch.tensor(0)
            if hasattr(module, 'weight'):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                if module.weight.requires_grad:
                    summary[m_key]['trainable'] = True
                else:
                    summary[m_key]['trainable'] = False
            if hasattr(module, 'bias') and module.bias is not None:
                params +=  torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]['nb_params'] = params

        if not isinstance(module, torch.nn.Sequential) and \
           not isinstance(module, torch.nn.ModuleList) and \
           not (module == model):
            hooks.append(module.register_forward_hook(hook))

    # check if there are multiple inputs to the network
    if isinstance(input_size[0], (list, tuple)):
        x = [torch.rand(3,*in_size) for in_size in input_size]
    else:
        x = torch.rand(3,*input_size)
    x = x.to(device)
    # create properties
    summary = OrderedDict()
    hooks = []
    # register hook
    model.apply(register_hook)
    # make a forward pass
    model(x)
    # remove these hooks
    for h in hooks:
        h.remove()

    print('----------------------------------------------------------------')
    line_new = '%-20s  %-25s %-15s'%('Layer (type)', 'Output Shpae', 'Param #')
    print(line_new)
    print('================================================================')
    total_params = torch.tensor(0)
    trainable_params = torch.tensor(0)
    for layer in summary:
        ## input_shape, output_shape, trainable, nb_params
        line_new = '%-20s  %-25s %-15s'%(layer, summary[layer]['output_shape'], summary[layer]['nb_params'].item())
        total_params += summary[layer]['nb_params']
        if 'trainable' in summary[layer]:
            if summary[layer]['trainable'] == True:
                trainable_params += summary[layer]['nb_params']
        print(line_new)
    print('================================================================')
    print('Total params: ' + str(total_params.item()))
    print('Trainable params: ' + str(trainable_params.item()))
    print('Non-trainable params: ' + str(total_params.item() - trainable_params.item()))
    print('----------------------------------------------------------------')    