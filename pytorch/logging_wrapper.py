from functools import wraps
import inspect
import collections
import torch

CALLED_METHOD = collections.defaultdict(bool)


def log_shape_dtype(method):
    @wraps(method)
    def _method_with_logging(*args, **kwargs):
        """Print shape and dtype information before and after func call"""
        outputs = method(*args, **kwargs)
        if not CALLED_METHOD[method]:
            print(f"Logging info for {method.__module__} {method.__qualname__} ...")
            method_args = inspect.signature(method).bind(*args, **kwargs).arguments
            for (var_name, value) in method_args.items():
                if var_name == "self":
                    continue
                if isinstance(value, torch.Tensor) or not isinstance(
                    value, collections.Iterable
                ):
                    print(
                        f"{method.__qualname__} Input {var_name!r} , shape = {list(value.shape)}, dtype = {value.dtype}"
                    )
                else:
                    for atom_value in value:
                        print(
                            f"{method.__qualname__} Input {var_name!r} , shape = {list(atom_value.shape)}, dtype = {atom_value.dtype}"
                        )
            log_outputs = outputs
            if isinstance(log_outputs, torch.Tensor) or not isinstance(
                log_outputs, collections.Iterable
            ):
                log_outputs = (log_outputs,)
            for idx, _out in enumerate(log_outputs):
                if isinstance(_out, torch.Tensor) or not isinstance(
                    _out, collections.Iterable
                ):
                    print(
                        f"{method.__qualname__} output {idx+1} , shape = {list(_out.shape)}, dtype = {_out.dtype}"
                    )
                else:
                    for sub_idx, atom_out in enumerate(_out):
                        print(
                            f"{method.__qualname__} output {idx+1} suboutput {sub_idx+1} , shape = {list(atom_value.shape)}, dtype = {atom_value.dtype}"
                        )

            print(f"Done logging for {method.__module__} {method.__qualname__} ")
            CALLED_METHOD[method] = True
        return outputs

    return _method_with_logging

