"""by lyuwenyu"""

from typing import Dict

from .det_solver import DetSolver
from .solver import BaseSolver

# key: the (single) task name, or `multitask` if the config lists several tasks
TASKS: Dict[str, BaseSolver] = {
    "detection": DetSolver,
    "body": DetSolver,
    "oih": DetSolver,
    "multitask": DetSolver,
}
