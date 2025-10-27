from gemlite.helper import *

def patch_model(model, device, processor, skip_modules=[]):
    #Loadd HQQLinear when needed
    if(processor in [A16Wn]):
        from hqq.core.quantize import HQQLinear
    else:
        class _NoHQQ: pass
        HQQLinear = _NoHQQ

    #Name modules
    for name, module in model.named_modules():
        module.name = name

    #Patching fct
    def _patching_fct(layer, device, skip_modules):
        layer = layer.to(device, non_blocking=True)
        if(any(s in layer.name for s in skip_modules)):
            return layer
        else:
            if(isinstance(layer, torch.nn.Linear)):
                return processor(device=device).from_linear(layer)
            elif(isinstance(layer, HQQLinear)):
                return processor(device=device).from_hqqlinear(layer)
            else:
                return layer

    #Replaces linear layers
    def _patch_linearlayers(model, fct, device, skip_modules):
        for name, layer in model.named_children():
            if isinstance(layer, (torch.nn.Linear, HQQLinear)):
                setattr(model, name, fct(layer, device, skip_modules))
            else:
                _patch_linearlayers(layer, fct, device, skip_modules)

    #Apply patch
    _patch_linearlayers(model, _patching_fct, device, skip_modules)

    #Clean-up
    torch.cuda.empty_cache()
    gc.collect()



patch_model(model, 'cuda:0', processor=A8W8_INT8_dynamic, skip_modules=['lm_head'])

