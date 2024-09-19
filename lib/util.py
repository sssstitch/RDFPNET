import atexit
import enum
import json
import os
import pickle
import shutil
import time
import uuid
from copy import deepcopy
from dataclasses import fields, is_dataclass
from pathlib import Path
#from pprint import pprint
from typing import Any, Optional, TypeVar, Union, cast, Dict, Tuple, Type
try:
    from typing import get_args, get_origin
except ImportError:
    from typing_extensions import get_args, get_origin

import numpy as np
import tomli
import tomli_w
import torch


from . import env

RawConfig = Dict[str, Any]
Report = Dict[str, Any]
T = TypeVar('T')


class Part(enum.Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'

    def __str__(self) -> str:
        return self.value


class TaskType(enum.Enum):
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'

    def __str__(self) -> str:
        return self.value




def update_training_log(training_log, data, metrics):
    def _update(log_part, data_part):
        for k, v in data_part.items():
            if isinstance(v, dict):
                _update(log_part.setdefault(k, {}), v)
            elif isinstance(v, list):
                log_part.setdefault(k, []).extend(v)
            else:
                log_part.setdefault(k, []).append(v)

    _update(training_log, data)
    transposed_metrics = {}
    for part, part_metrics in metrics.items():
        for metric_name, value in part_metrics.items():
            transposed_metrics.setdefault(metric_name, {})[part] = value
    _update(training_log, transposed_metrics)


def raise_unknown(unknown_what: str, unknown_value: Any):
    raise ValueError(f'Unknown {unknown_what}: {unknown_value}')


def _replace(data, condition, value):
    def do(x):
        if isinstance(x, dict):
            return {k: do(v) for k, v in x.items()}
        elif isinstance(x, list):
            return [do(y) for y in x]
        else:
            return value if condition(x) else x

    return do(data)


_CONFIG_NONE = '__none__'


def unpack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x == _CONFIG_NONE, None))
    return config


def pack_config(config: RawConfig) -> RawConfig:
    config = cast(RawConfig, _replace(config, lambda x: x is None, _CONFIG_NONE))
    return config


def load_config(path: Union[Path, str]) -> Any:
    with open(path, 'rb') as f:
        return unpack_config(tomli.load(f))


def dump_config(config: Any, path: Union[Path, str]) -> None:
    with open(path, 'wb') as f:
        tomli_w.dump(pack_config(config), f)
    assert config == load_config(path)


def load_json(path: Union[Path, str], **kwargs) -> Any:
    return json.loads(Path(path).read_text(), **kwargs)


def dump_json(x: Any, path: Union[Path, str], **kwargs) -> None:
    kwargs.setdefault('indent', 4)
    Path(path).write_text(json.dumps(x, **kwargs) + '\n')


def load_pickle(path: Union[Path, str], **kwargs) -> Any:
    return pickle.loads(Path(path).read_bytes(), **kwargs)


def dump_pickle(x: Any, path: Union[Path, str], **kwargs) -> None:
    Path(path).write_bytes(pickle.dumps(x, **kwargs))


def load(path: Union[Path, str], **kwargs) -> Any:
    return globals()[f'load_{Path(path).suffix[1:]}'](Path(path), **kwargs)


def dump(x: Any, path: Union[Path, str], **kwargs) -> Any:
    return globals()[f'dump_{Path(path).suffix[1:]}'](x, Path(path), **kwargs)


def _get_output_item_path(
    path: Union[str, Path], filename: str, must_exist: bool
) -> Path:
    path = env.get_path(path)
    if path.suffix == '.toml':
        path = path.with_suffix('')
    if path.is_dir():
        path = path / filename
    else:
        assert path.name == filename
    assert path.parent.exists()
    if must_exist:
        assert path.exists()
    return path


def load_report(path: Path) -> Report:
    return load_json(_get_output_item_path(path, 'report.json', True))


def dump_report(report: dict, path: Path) -> None:
    dump_json(report, _get_output_item_path(path, 'report.json', False))


def load_predictions(path: Path) -> Dict[str, np.ndarray]:
    with np.load(_get_output_item_path(path, 'predictions.npz', True)) as predictions:
        return {x: predictions[x] for x in predictions}


def dump_predictions(predictions: Dict[str, np.ndarray], path: Path) -> None:
    np.savez(_get_output_item_path(path, 'predictions.npz', False), **predictions)


def dump_metrics(metrics: Dict[str, Any], path: Path) -> None:
    dump_json(metrics, _get_output_item_path(path, 'metrics.json', False))


def load_checkpoint(path: Path, *args, **kwargs) -> Dict[str, np.ndarray]:
    return torch.load(
        _get_output_item_path(path, 'checkpoint.pt', True), *args, **kwargs
    )


def get_device() -> torch.device:
    if torch.cuda.is_available():
        assert os.environ.get('CUDA_VISIBLE_DEVICES') is not None
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


def _print_sep(c, size=100):
    print(c * size)


_LAST_SNAPSHOT_TIME = None


def backup_output(output_dir: Path) -> None:
    backup_dir = os.environ.get('TMP_OUTPUT_PATH')
    snapshot_dir = os.environ.get('SNAPSHOT_PATH')
    if backup_dir is None:
        assert snapshot_dir is None
        return
    assert snapshot_dir is not None

    try:
        relative_output_dir = output_dir.relative_to(env.PROJ)
    except ValueError:
        return

    for dir_ in [backup_dir, snapshot_dir]:
        new_output_dir = dir_ / relative_output_dir
        prev_backup_output_dir = new_output_dir.with_name(new_output_dir.name + '_prev')
        new_output_dir.parent.mkdir(exist_ok=True, parents=True)
        if new_output_dir.exists():
            new_output_dir.rename(prev_backup_output_dir)
        shutil.copytree(output_dir, new_output_dir)
        if output_dir.with_suffix('.toml').exists():
            shutil.copyfile(
                output_dir.with_suffix('.toml'), new_output_dir.with_suffix('.toml')
            )
        if prev_backup_output_dir.exists():
            shutil.rmtree(prev_backup_output_dir)

    global _LAST_SNAPSHOT_TIME
    if _LAST_SNAPSHOT_TIME is None or time.time() - _LAST_SNAPSHOT_TIME > 10 * 60:
        import nirvana_dl.snapshot

        nirvana_dl.snapshot.dump_snapshot()
        _LAST_SNAPSHOT_TIME = time.time()
        print('The snapshot was saved!')


def _get_scores(metrics: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, float]]:
    return (
        {k: v['score'] for k, v in metrics.items()}
        if 'score' in next(iter(metrics.values()))
        else None
    )


def format_scores(metrics: Dict[str, Dict[str, Any]]) -> str:
    return ' '.join(
        f"[{x}] {metrics[x]['score']:.3f}"
        for x in ['test', 'val', 'train']
        if x in metrics
    )


def finish(output_dir: Path, report: dict) -> None:
    print()
    _print_sep('=')

    metrics = report.get('metrics')
    if metrics is not None:
        scores = _get_scores(metrics)
        if scores is not None:
            dump_json(scores, output_dir / 'scores.json')
            print(format_scores(metrics))
            _print_sep('-')

    dump_report(report, output_dir)
    json_output_path = os.environ.get('JSON_OUTPUT_FILE')
    if json_output_path:
        try:
            key = str(output_dir.relative_to(env.PROJ))
        except ValueError:
            pass
        else:
            json_output_path = Path(json_output_path)
            try:
                json_data = json.loads(json_output_path.read_text())
            except (FileNotFoundError, json.decoder.JSONDecodeError):
                json_data = {}
            json_data[key] = load_json(output_dir / 'report.json')
            json_output_path.write_text(json.dumps(json_data, indent=4))
        shutil.copyfile(
            json_output_path,
            os.path.join(os.environ['SNAPSHOT_PATH'], 'json_output.json'),
        )

    output_dir.joinpath('DONE').touch()
    backup_output(output_dir)
    print(f'Done! | {report.get("time")} | {output_dir}')
    _print_sep('=')
    print()


def from_dict(datacls: Type[T], data: dict) -> T:
    assert is_dataclass(datacls)
    data = deepcopy(data)
    for field in fields(datacls):
        if field.name not in data:
            continue
        if is_dataclass(field.type):
            data[field.name] = from_dict(field.type, data[field.name])
        elif (
            get_origin(field.type) is Union
            and len(get_args(field.type)) == 2
            and get_args(field.type)[1] is type(None)
            and is_dataclass(get_args(field.type)[0])
        ):
            if data[field.name] is not None:
                data[field.name] = from_dict(get_args(field.type)[0], data[field.name])
    return datacls(**data)


def replace_factor_with_value(
    config: RawConfig,
    key: str,
    reference_value: int,
    bounds: Tuple[float, float],
) -> None:
    factor_key = key + '_factor'
    if factor_key not in config:
        assert key in config
    else:
        assert key not in config
        factor = config.pop(factor_key)
        assert bounds[0] <= factor <= bounds[1]
        config[key] = int(factor * reference_value)


def get_temporary_copy(path: Union[str, Path]) -> Path:
    path = env.get_path(path)
    assert not path.is_dir() and not path.is_symlink()
    tmp_path = path.with_name(
        path.stem + '___' + str(uuid.uuid4()).replace('-', '') + path.suffix
    )
    shutil.copyfile(path, tmp_path)
    atexit.register(lambda: tmp_path.unlink())
    return tmp_path
