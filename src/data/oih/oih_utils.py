from typing import Any, Dict, List, Optional

import torch
from torch import Tensor


class BaseTensorStruct:
    def __init__(self, **kwargs):
        self._fields = kwargs

    def to(self, *args, **kwargs):
        return BaseTensorStruct(**{k: v.to(*args, **kwargs) for k, v in self._fields.items()})

    def detach(self):
        return BaseTensorStruct(**{k: v.detach() for k, v in self._fields.items()})

    def clone(self):
        return BaseTensorStruct(**{k: v.clone() for k, v in self._fields.items()})

    def __repr__(self):
        return f"Tensor_Struct {', '.join(k for k, v in self._fields.items())}"

    def get(self, key: str, default: Optional[Any] = None):
        return self._fields.get(key, default)

    def set(self, key: str, value: Tensor):
        self._fields[key] = value

    def set_dict(self, dct: Dict[str, Tensor]):
        for k, v in dct.items():
            self.set(k, v)
        return self


class OiHStruct(BaseTensorStruct):
    """Parses a similarity-based (``oih_sim``) OiH tensor into named fields.

    Layout per hand: [vis (optional)] cls box(4) obj_box(4) obj_cls(1); right hand first.
    """

    def __init__(self, tensor: Tensor, **kwargs):
        assert "oih_cls" in kwargs
        assert kwargs.get("oih_sim", False), "Only oih_sim OiH tensors are supported."

        oih_cls = kwargs["oih_cls"]
        self.oih_vis = kwargs.get("oih_vis", False)
        self.oih_sim = True
        obj_cls_dim = 1
        oih_vis_dim = 1 if self.oih_vis else 0

        fields = {
            "right_hand_vis": tensor[..., :oih_vis_dim],  # :1
            "right_hand_cls": tensor[..., oih_vis_dim : oih_cls + oih_vis_dim],  # 1 : 3
            "right_hand_box": tensor[
                ..., oih_cls + oih_vis_dim : oih_cls + 4 + oih_vis_dim
            ],  # 3 : 7
            "right_hand_obj_box": tensor[
                ..., (oih_cls + 4 + oih_vis_dim) : (oih_cls + 4 * 2 + oih_vis_dim)
            ],  # 7 : 11
            "right_hand_obj_cls": tensor[
                ..., (oih_cls + 4 * 2 + oih_vis_dim) : (oih_cls + 4 * 2 + oih_vis_dim + obj_cls_dim)
            ],  # 11 : 12
            "left_hand_vis": tensor[
                ...,
                (oih_cls + 4 * 2 + oih_vis_dim + obj_cls_dim) : (
                    oih_cls + 4 * 2 + oih_vis_dim * 2 + obj_cls_dim
                ),
            ],  # 12 : 13
            "left_hand_cls": tensor[
                ...,
                (oih_cls + 4 * 2 + oih_vis_dim * 2 + obj_cls_dim) : (
                    oih_cls * 2 + 4 * 2 + oih_vis_dim * 2 + obj_cls_dim
                ),
            ],  # 13 : 15
            "left_hand_box": tensor[
                ...,
                (oih_cls * 2 + 4 * 2 + oih_vis_dim * 2 + obj_cls_dim) : (
                    oih_cls * 2 + 4 * 3 + oih_vis_dim * 2 + obj_cls_dim
                ),
            ],  # 15 : 19
            "left_hand_obj_box": tensor[
                ..., oih_cls * 2 + 4 * 3 + oih_vis_dim * 2 + obj_cls_dim : -obj_cls_dim
            ],  # 19:23
            "left_hand_obj_cls": tensor[..., -obj_cls_dim:],  # last dim
        }
        self.tensor = tensor
        self.shape = tensor.shape
        # filtering shape[-1] == 0 tensor(*_vis tensor may filtered)
        fields = {k: v for k, v in fields.items() if v.shape[-1] != 0}
        super(OiHStruct, self).__init__(**fields)

    def __getitem__(self, item: str) -> Tensor:
        assert isinstance(item, str), f"OiH Structure cannot subscribe {type(item)}. Using str"
        return self._fields.get(item, None)

    def serialize(self) -> Tensor:
        """Returns: serialized tensor"""
        return torch.cat([v for k, v in self._fields.items()], dim=-1)

    def add(self, kv_pair: Dict) -> Dict:
        for k, v in kv_pair.items():
            if k in self._fields:
                self._fields[k] = self._fields[k] + v
        return self._fields

    def add_(self, kv_pair: Dict) -> Dict:
        for k, v in kv_pair.items():
            if k in self._fields:
                self._fields[k].add_(v)
        return self._fields

    def execute(self, stub: dict) -> None:
        """stub: {key: func} or {key: (func, arg)}"""
        for k, v in stub.items():
            assert k in self._fields, f"Key {k} not found in OiHStruct: {self._fields.keys()}"
            if isinstance(v, List):
                self._fields[k] = v[0](self._fields[k], v[1])
            else:
                self._fields[k] = v(self._fields[k])

    def get_boxes(self) -> Dict:
        return {
            k: self.get(k)
            for k in ("right_hand_box", "left_hand_box", "right_hand_obj_box", "left_hand_obj_box")
        }
