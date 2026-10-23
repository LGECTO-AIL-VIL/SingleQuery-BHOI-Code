"""Category metadata.

Only category list 1.9 is provided: the 80 COCO thing classes plus the
class-agnostic "object" class (dataset id 240) that is used as the
interaction-object target. All 81 classes are things.
"""

from functools import lru_cache
from types import SimpleNamespace

__all__ = ["get_metadata"]

# (dataset category id, name); the contiguous id is the position in the list.
CATEGORIES_1_9 = [
    (1, "person"),
    (2, "bicycle"),
    (3, "car"),
    (4, "motorcycle"),
    (5, "airplane"),
    (6, "bus"),
    (7, "train"),
    (8, "truck"),
    (9, "boat"),
    (10, "traffic light"),
    (11, "fire hydrant"),
    (13, "stop sign"),
    (14, "parking meter"),
    (15, "bench"),
    (16, "bird"),
    (17, "cat"),
    (18, "dog"),
    (19, "horse"),
    (20, "sheep"),
    (21, "cow"),
    (22, "elephant"),
    (23, "bear"),
    (24, "zebra"),
    (25, "giraffe"),
    (27, "backpack"),
    (28, "umbrella"),
    (31, "handbag"),
    (32, "tie"),
    (33, "suitcase"),
    (34, "frisbee"),
    (35, "skis"),
    (36, "snowboard"),
    (37, "sports ball"),
    (38, "kite"),
    (39, "baseball bat"),
    (40, "baseball glove"),
    (41, "skateboard"),
    (42, "surfboard"),
    (43, "tennis racket"),
    (44, "bottle"),
    (46, "wine glass"),
    (47, "cup"),
    (48, "fork"),
    (49, "knife"),
    (50, "spoon"),
    (51, "bowl"),
    (52, "banana"),
    (53, "apple"),
    (54, "sandwich"),
    (55, "orange"),
    (56, "broccoli"),
    (57, "carrot"),
    (58, "hot dog"),
    (59, "pizza"),
    (60, "donut"),
    (61, "cake"),
    (62, "chair"),
    (63, "couch"),
    (64, "potted plant"),
    (65, "bed"),
    (67, "dining table"),
    (70, "toilet"),
    (72, "tv"),
    (73, "laptop"),
    (74, "mouse"),
    (75, "remote"),
    (76, "keyboard"),
    (77, "cell phone"),
    (78, "microwave"),
    (79, "oven"),
    (80, "toaster"),
    (81, "sink"),
    (82, "refrigerator"),
    (84, "book"),
    (85, "clock"),
    (86, "vase"),
    (87, "scissors"),
    (88, "teddy bear"),
    (89, "hair drier"),
    (90, "toothbrush"),
    (240, "object"),
]


@lru_cache(maxsize=None)
def get_metadata(category_ver):
    """Return the category metadata of `category_ver` (only "1.9" is supported)."""
    if str(category_ver) != "1.9":
        raise ValueError(f"Unsupported category version: {category_ver}")

    thing_classes = [name for _, name in CATEGORIES_1_9]
    thing_dataset_id_to_contiguous_id = {cid: i for i, (cid, _) in enumerate(CATEGORIES_1_9)}
    return SimpleNamespace(
        thing_classes=thing_classes,
        thing_categories=[dict(id=cid, name=name, isthing=1) for cid, name in CATEGORIES_1_9],
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        contiguous_id_to_dataset_id={
            i: cid for cid, i in thing_dataset_id_to_contiguous_id.items()
        },
    )
