import glob
import json
import os
from typing import Callable, Dict, Iterable, Literal, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.core import register

from .oih_data_utils import oih_data_loader

__all__ = ["ContactHands"]


@register
class ContactHands(Dataset):
    __inject__ = ["transforms", "evaluators"]
    __share__ = [
        "things_only",
        "person_only",
        "oih_config",
    ]

    def __init__(
        self,
        mode: Literal["train", "val"],
        img_folder: str,
        gt_folder: str,
        transforms: Optional[Callable],
        oih_config: Optional[Dict] = None,
        evaluators: Optional[Iterable] = None,
        things_only: bool = False,
        person_only: bool = False,
        use_load_cache: bool = False,
    ) -> None:
        assert mode == "val", 'Contact Hands evalset is only support "val" mode'
        self.mode = mode
        self.img_folder = img_folder
        self.gt_folder = gt_folder
        self.anno_files = list(glob.glob(os.path.join(self.gt_folder, "*.json")))  # for testing
        self.things_only = things_only
        self.person_only = person_only
        self._transforms = transforms
        self.cost = ["oih"]
        self.evaluators = evaluators
        self.dataset_name = "ContactHands"
        assert oih_config is not None, (
            '(OIH) Need oih_config. with "oih_cls:int", "oih_contact:bool"'
        )
        self.oih_config = oih_config
        self.oih_loader = oih_data_loader(oih_config["oih_cls"], oih_config["oih_contact"])

        if use_load_cache:
            self.img_reader = CachedReader(ImReader(), cache_dict=None)
            self.anno_reader = CachedReader(self.read_anno, cache_dict=None)
        else:
            self.img_reader = ImReader()
            self.anno_reader = self.read_anno

        if self.person_only:
            person_only_idxs = {}
            # person_category_id = self.metadata.contiguous_id_to_dataset_id[
            #     self.metadata.stuff_classes.index("person")
            # ]
            # TODO : read from metadata. above code willl..
            person_category_id = 0  # pseudo labeling..
            for idx, path in enumerate(self.anno_files):
                anno = self.anno_reader(path)
                has_human = False
                for obj in anno["annotations"][0]["segments_info"]:
                    if obj["category_id"] == person_category_id and "keypoints" in obj:
                        has_human = True

                    #     has_human = True
                if has_human:
                    person_only_idxs[idx] = 1
            self.idxs = person_only_idxs
            self.anno_files = [
                annofile for i, annofile in enumerate(self.anno_files) if i in self.idxs
            ]

        assert len(self.anno_files) != 0, f"(OIH) Nothing in {self.gt_folder}. check path"
        print(f"(OIH)Contact hands data len: {len(self.anno_files)}")

    # need to change static func?
    def read_anno(self, fpath: str) -> Dict:
        with open(fpath) as ff:
            return json.load(ff)

    def postprocess(
        self,
        sample: Dict,
    ) -> Dict:
        return sample

    def get_height_and_width(self, idx: int) -> Tuple[int, int]:
        anno = self.anno_reader(self.anno_files[idx])
        width = anno["images"][0]["width"]
        height = anno["images"][0]["height"]
        return height, width

    def __getitem__(
        self,
        idx: int,
    ) -> Tuple[torch.Tensor, Dict]:
        anno = self.anno_reader(self.anno_files[idx])
        width = anno["images"][0]["width"]
        height = anno["images"][0]["height"]

        filename = anno["images"][0]["file_name"]
        img_filepath = os.path.join(self.img_folder, filename)

        image = self.img_reader(img_filepath).convert("RGB")
        img_w, img_h = image.size  # real size

        ## check size in anno file and real image
        #
        if width != img_w or height != img_h:
            image = image.resize((width, height), resample=Image.LANCZOS)

        if self.mode == "train":
            # can not reach here
            pass
        else:
            target = {
                "image_id": filename,
                "size": torch.tensor([width, height]),
                "orig_size": torch.tensor([width, height]),
            }

            anno = self.anno_reader(self.anno_files[idx])
            anno = anno["annotations"][0]["segments_info"]
            oih = []
            oih_cont = []
            valid_oih = []
            oih_boxes = []
            oih_obj_boxes = []
            oih_visibility = []

            boxes = []
            area = []
            labels = []
            for ins in anno:
                (oih_ins, oih_box, oih_obj_box, oih_cont_ins, valid_oih_ins, oih_vis, oih_ids) = (
                    self.oih_loader.load(ins)
                )
                oih_boxes.append(oih_box)
                oih_obj_boxes.append(oih_obj_box)
                oih.append(oih_ins)
                oih_cont.append(oih_cont_ins)
                valid_oih.append(valid_oih_ins)
                # xywh to xyxy in bbox
                ins["bbox"][3] += ins["bbox"][1]
                ins["bbox"][2] += ins["bbox"][0]
                boxes.append(ins["bbox"])
                area.append(ins["area"])
                labels.append(ins["category_id"])
                oih_visibility.append(oih_vis)

            target["boxes"] = torch.tensor(np.array(boxes), dtype=torch.float)
            target["area"] = torch.tensor(area)
            target["labels"] = torch.tensor(labels)
            target["oih_vis"] = torch.tensor(np.array(oih_visibility), dtype=torch.float)

            oih_boxes = torch.tensor(np.array(oih_boxes), dtype=torch.float)
            target["oih_boxes"] = oih_boxes
            oih_obj_boxes = torch.tensor(np.array(oih_obj_boxes), dtype=torch.float)
            target["oih_obj_boxes"] = oih_obj_boxes
            oih = torch.tensor(np.array(oih), dtype=torch.float)
            target["oih"] = oih
            oih_cont = torch.tensor(np.array(oih_cont), dtype=torch.float)
            target["oih_cont"] = oih_cont
            target["valid_oih"] = torch.tensor(valid_oih).bool()
            if target["valid_oih"].sum() != 0:
                target["cost"] = self.cost

            if self._transforms:
                image, target = self._transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.anno_files)

    def get_evaluators(self) -> Iterable:
        return self.evaluators


class CachedReader:
    def __init__(self, reader: Callable, cache_dict: Dict = None) -> None:
        """
        read using reader with cache(using  reader's param)
        Args:
            reader: read func or callable obj
            cache_dict: cache. if None, save in instance
        """
        self.reader = reader
        self._cache = cache_dict if cache_dict is not None else {}

    def open(
        self,
        fpath: str,
    ) -> object:
        im = self._cache.get(fpath, None)
        if im is None:
            im = self._cache[fpath] = self.reader(fpath)
        return im

    def __call__(self, fpath: str) -> object:
        return self.open(fpath)


class ImReader:
    def __init__(self) -> None:
        pass

    def open(
        self,
        fpath: str,
    ) -> Image.Image:
        return Image.open(fpath)

    def __call__(self, fpath: str) -> Image.Image:
        return self.open(fpath)
