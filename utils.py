from datetime import datetime, timedelta
import time
import numpy as np
from typing import Callable
from lmfit import Model

T = float | np.ndarray

class LMFit:
    __slots__ = ('f', 'parameters')

    def __init__(self, f: Callable[[T], T]):
        self.f = f
        self.parameters: dict[str, tuple[float, float]] = {}

    def __getattr__(self, variable: str) -> float:
        return self.parameters[variable]

    @property
    def args(self) -> tuple[str]:
        return self.f.__code__.co_varnames[1:]  # assuming first argument is x

    def fit(self, x: list[float], y: list[float], **kws) -> None:
        model = Model(self.f)
        initial = {arg: 1 for arg in self.args}
        initial.update(kws)
        result = model.fit(y, x=x, **initial)
        self.parameters.update({p: (v.value, v.stderr) for p, v in result.params.items()})

    def get(self, start: float, end: float, n: int = 300) -> tuple[np.ndarray, np.ndarray]:
        x = np.linspace(start, end, n)
        y = self.f(x, **{p: v[0] for p, v in self.parameters.items()})
        return x, y

class Progress:
    def __init__(self, fmt: str = '[{dtime:%H:%M:%S}] {bar} | {percent:.2f}% | {elapsed} | {remaining}', bar_length: int = 30) -> None:
        self.fmt = fmt
        self._checkpoints: list[float] = []
        self._start: float = None

        def progressbar(progress: float) -> str:
            quarters = '_░▒▓█'
            done = int(progress * bar_length)
            return (done * '█' + quarters[round(4 * (bar_length * progress - done))] + int((1 - progress) * bar_length) * '_')[:bar_length]
        self.progressbar = progressbar

    def start(self) -> float:
        self._start = time.perf_counter()
        return self._start
    
    def checkpoint(self) -> float:
        perf = time.perf_counter()
        self._checkpoints.append(perf)
        return perf

    def string(self, progress: float, use_checkpoint: int = -1, print_string: bool = True, *, prefix: str = '', suffix: str = '', **kwargs) -> str:
        delta = time.perf_counter() - (self._start if use_checkpoint == -1 else self._checkpoints[use_checkpoint])
        string = self.fmt.format(bar=self.progressbar(progress),
                                 percent=100 * progress,
                                 dtime=datetime.now(),
                                 elapsed=str(timedelta(seconds=round(delta))),
                                 remaining=str(timedelta(seconds=0 if progress == 0.0 else round(delta/progress))),
                                 **kwargs)
        if prefix != '':
            string = f'{prefix} | ' + string
        if suffix != '':
            string += f' | {suffix}'

        if print_string:
            print(f'\r{string}', end="\n" if progress == 1.0 else "")
        return string

def linmap(x: T, from_range: tuple[T, T], to_range: tuple[T, T]) -> T:
    return to_range[0] + (to_range[1] - to_range[0]) * (x - from_range[0])/(from_range[1] - from_range[0])

def expmap(x: T, from_range: tuple[T, T], to_range: tuple[T, T], base: float = np.e) -> T:
    return to_range[0] * (to_range[1] / to_range[0])**((x - from_range[0])/(from_range[1] - from_range[0]) * np.log(base))

def progressbar(bar_length: int = 30) -> Callable[[float], str]:
    def f(progress: float):
        quarters = '_░▒▓█'
        done = int(progress * bar_length)
        return (done * '█' + quarters[round(4 * (bar_length * progress - done))] + int((1 - progress) * bar_length) * '_')[:bar_length]
    return f
