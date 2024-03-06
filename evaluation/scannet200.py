import numpy as np
import torch

import os.path as osp
from .custom import CustomDataset


CLASS_LABELS_200 = (
    "wall",
    "chair",
    "floor",
    "table",
    "door",
    "couch",
    "cabinet",
    "shelf",
    "desk",
    "office chair",
    "bed",
    "pillow",
    "sink",
    "picture",
    "window",
    "toilet",
    "bookshelf",
    "monitor",
    "curtain",
    "book",
    "armchair",
    "coffee table",
    "box",
    "refrigerator",
    "lamp",
    "kitchen cabinet",
    "towel",
    "clothes",
    "tv",
    "nightstand",
    "counter",
    "dresser",
    "stool",
    "cushion",
    "plant",
    "ceiling",
    "bathtub",
    "end table",
    "dining table",
    "keyboard",
    "bag",
    "backpack",
    "toilet paper",
    "printer",
    "tv stand",
    "whiteboard",
    "blanket",
    "shower curtain",
    "trash can",
    "closet",
    "stairs",
    "microwave",
    "stove",
    "shoe",
    "computer tower",
    "bottle",
    "bin",
    "ottoman",
    "bench",
    "board",
    "washing machine",
    "mirror",
    "copier",
    "basket",
    "sofa chair",
    "file cabinet",
    "fan",
    "laptop",
    "shower",
    "paper",
    "person",
    "paper towel dispenser",
    "oven",
    "blinds",
    "rack",
    "plate",
    "blackboard",
    "piano",
    "suitcase",
    "rail",
    "radiator",
    "recycling bin",
    "container",
    "wardrobe",
    "soap dispenser",
    "telephone",
    "bucket",
    "clock",
    "stand",
    "light",
    "laundry basket",
    "pipe",
    "clothes dryer",
    "guitar",
    "toilet paper holder",
    "seat",
    "speaker",
    "column",
    "bicycle",
    "ladder",
    "bathroom stall",
    "shower wall",
    "cup",
    "jacket",
    "storage bin",
    "coffee maker",
    "dishwasher",
    "paper towel roll",
    "machine",
    "mat",
    "windowsill",
    "bar",
    "toaster",
    "bulletin board",
    "ironing board",
    "fireplace",
    "soap dish",
    "kitchen counter",
    "doorframe",
    "toilet paper dispenser",
    "mini fridge",
    "fire extinguisher",
    "ball",
    "hat",
    "shower curtain rod",
    "water cooler",
    "paper cutter",
    "tray",
    "shower door",
    "pillar",
    "ledge",
    "toaster oven",
    "mouse",
    "toilet seat cover dispenser",
    "furniture",
    "cart",
    "storage container",
    "scale",
    "tissue box",
    "light switch",
    "crate",
    "power outlet",
    "decoration",
    "sign",
    "projector",
    "closet door",
    "vacuum cleaner",
    "candle",
    "plunger",
    "stuffed animal",
    "headphones",
    "dish rack",
    "broom",
    "guitar case",
    "range hood",
    "dustpan",
    "hair dryer",
    "water bottle",
    "handicap bar",
    "purse",
    "vent",
    "shower floor",
    "water pitcher",
    "mailbox",
    "bowl",
    "paper bag",
    "alarm clock",
    "music stand",
    "projector screen",
    "divider",
    "laundry detergent",
    "bathroom counter",
    "object",
    "bathroom vanity",
    "closet wall",
    "laundry hamper",
    "bathroom stall door",
    "ceiling light",
    "trash bin",
    "dumbbell",
    "stair rail",
    "tube",
    "bathroom cabinet",
    "cd case",
    "closet rod",
    "coffee kettle",
    "structure",
    "shower head",
    "keyboard piano",
    "case of water bottles",
    "coat rack",
    "storage organizer",
    "folded chair",
    "fire alarm",
    "power strip",
    "calendar",
    "poster",
    "potted plant",
    "luggage",
    "mattress",
)

# CLASSES = [CLASS_LABELS_200[i] for i in BENCHMARK_SEMANTIC_IDXS[2:]]


class ScanNet200Dataset(CustomDataset):
    # BENCHMARK_SEMANTIC_IDXS = np.load("dataset/scannet200/reverse_norm_ids.npy")

    BENCHMARK_SEMANTIC_IDXS = [
        1,
        3,
        2,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        21,
        22,
        23,
        24,
        26,
        27,
        28,
        29,
        31,
        32,
        33,
        34,
        35,
        36,
        38,
        39,
        40,
        41,
        42,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        54,
        55,
        56,
        57,
        58,
        59,
        62,
        63,
        64,
        65,
        66,
        67,
        68,
        69,
        70,
        71,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        82,
        84,
        86,
        87,
        88,
        89,
        90,
        93,
        95,
        96,
        97,
        98,
        99,
        100,
        101,
        102,
        103,
        104,
        105,
        106,
        107,
        110,
        112,
        115,
        116,
        118,
        120,
        121,
        122,
        125,
        128,
        130,
        131,
        132,
        134,
        136,
        138,
        139,
        140,
        141,
        145,
        148,
        154,
        155,
        156,
        157,
        159,
        161,
        163,
        165,
        166,
        168,
        169,
        170,
        177,
        180,
        185,
        188,
        191,
        193,
        195,
        202,
        208,
        213,
        214,
        221,
        229,
        230,
        232,
        233,
        242,
        250,
        261,
        264,
        276,
        283,
        286,
        300,
        304,
        312,
        323,
        325,
        331,
        342,
        356,
        370,
        392,
        395,
        399,
        408,
        417,
        488,
        540,
        562,
        570,
        572,
        581,
        609,
        748,
        776,
        1156,
        1163,
        1164,
        1165,
        1166,
        1167,
        1168,
        1169,
        1170,
        1171,
        1172,
        1173,
        1174,
        1175,
        1176,
        1178,
        1179,
        1180,
        1181,
        1182,
        1183,
        1184,
        1185,
        1186,
        1187,
        1188,
        1189,
        1190,
        1191,
    ]

    SCANNET20_CLASSES = (
        "cabinet","bed","chair","sofa","table","door",
        "window","bookshelf","picture","counter","desk",
        "curtain","refrigerator","shower curtain","toilet",
        "sink","bathtub","otherfurniture",
    )
    BASE_CLASSES = [
        'chair', 'table', 'door', 'couch', 'cabinet', 'shelf', 'desk', 'office chair', 'bed', \
        'sink', 'picture', 'window', 'toilet', 'bookshelf', 'curtain', 'armchair', 'coffee table',\
        'refrigerator', 'kitchen cabinet', 'counter', 'dresser', 'ceiling', 'bathtub', 'end table', \
        'dining table', 'shower curtain', 'closet', 'ottoman', 'bench', 'sofa chair', 'file cabinet',\
        'blinds', 'container', 'wardrobe', 'seat', 'column', 'shower wall', 'kitchen counter', 'mini \
        fridge', 'shower door', 'pillar', 'furniture', 'storage container', 'closet door', \
        'shower floor', 'bathroom counter', 'closet wall', 'bathroom stall door', 'bathroom cabinet', \
        'folded chair', 'mattress'
    ]

    NOVEL_CLASSES = ['pillow', 'monitor', 'book', 'box', 'lamp', 'towel', 'clothes', 'tv', 'nightstand',\
                    'stool', 'cushion', 'plant', 'keyboard', 'bag', 'backpack', 'toilet paper', 'printer',\
                    'tv stand', 'whiteboard', 'blanket', 'trash can', 'stairs', 'microwave', 'stove', \
                    'shoe', 'computer tower', 'bottle', 'bin', 'board', 'washing machine', 'mirror',\
                    'copier', 'basket', 'fan', 'laptop', 'shower', 'paper', 'person',\
                    'paper towel dispenser', 'oven', 'rack', 'plate', 'blackboard', 'piano', 'suitcase',\
                    'rail', 'radiator', 'recycling bin', 'soap dispenser', 'telephone', 'bucket', 'clock', \
                    'stand', 'light', 'laundry basket', 'pipe', 'clothes dryer', 'guitar', \
                    'toilet paper holder', 'speaker', 'bicycle', 'ladder', 'bathroom stall', 
                    'cup', 'jacket', 'storage bin', 'coffee maker', 'dishwasher', 'paper towel roll', \
                    'machine', 'mat', 'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board', \
                    'fireplace', 'soap dish', 'doorframe', 'toilet paper dispenser', 'fire extinguisher',\
                    'ball', 'hat', 'shower curtain rod', 'water cooler', 'paper cutter', 'tray', \
                    'ledge', 'toaster oven', 'mouse', 'toilet seat cover dispenser', 'cart', 'scale',\
                    'tissue box', 'light switch', 'crate', 'power outlet', 'decoration', 'sign', \
                    'projector', 'vacuum cleaner', 'candle', 'plunger', 'stuffed animal', 'headphones', \
                    'dish rack', 'broom', 'guitar case', 'range hood', 'dustpan', 'hair dryer', \
                    'water bottle', 'handicap bar', 'purse', 'vent', 'water pitcher', 'mailbox', \
                    'bowl', 'paper bag', 'alarm clock', 'music stand', 'projector screen', 'divider', \
                    'laundry detergent', 'object', 'bathroom vanity', 'laundry hamper', 'ceiling light', \
                    'trash bin', 'dumbbell', 'stair rail', 'tube', 'cd case', 'closet rod', 'coffee kettle',\
                    'structure', 'shower head', 'keyboard piano', 'case of water bottles', 'coat rack',\
                    'storage organizer', 'fire alarm', 'power strip', 'calendar', 'poster', 'potted plant', \
                    'luggage']

    CLASSES = (
        "chair",
        "table",
        "door",
        "couch",
        "cabinet",
        "shelf",
        "desk",
        "office chair",
        "bed",
        "pillow",
        "sink",
        "picture",
        "window",
        "toilet",
        "bookshelf",
        "monitor",
        "curtain",
        "book",
        "armchair",
        "coffee table",
        "box",
        "refrigerator",
        "lamp",
        "kitchen cabinet",
        "towel",
        "clothes",
        "tv",
        "nightstand",
        "counter",
        "dresser",
        "stool",
        "cushion",
        "plant",
        "ceiling",
        "bathtub",
        "end table",
        "dining table",
        "keyboard",
        "bag",
        "backpack",
        "toilet paper",
        "printer",
        "tv stand",
        "whiteboard",
        "blanket",
        "shower curtain",
        "trash can",
        "closet",
        "stairs",
        "microwave",
        "stove",
        "shoe",
        "computer tower",
        "bottle",
        "bin",
        "ottoman",
        "bench",
        "board",
        "washing machine",
        "mirror",
        "copier",
        "basket",
        "sofa chair",
        "file cabinet",
        "fan",
        "laptop",
        "shower",
        "paper",
        "person",
        "paper towel dispenser",
        "oven",
        "blinds",
        "rack",
        "plate",
        "blackboard",
        "piano",
        "suitcase",
        "rail",
        "radiator",
        "recycling bin",
        "container",
        "wardrobe",
        "soap dispenser",
        "telephone",
        "bucket",
        "clock",
        "stand",
        "light",
        "laundry basket",
        "pipe",
        "clothes dryer",
        "guitar",
        "toilet paper holder",
        "seat",
        "speaker",
        "column",
        "bicycle",
        "ladder",
        "bathroom stall",
        "shower wall",
        "cup",
        "jacket",
        "storage bin",
        "coffee maker",
        "dishwasher",
        "paper towel roll",
        "machine",
        "mat",
        "windowsill",
        "bar",
        "toaster",
        "bulletin board",
        "ironing board",
        "fireplace",
        "soap dish",
        "kitchen counter",
        "doorframe",
        "toilet paper dispenser",
        "mini fridge",
        "fire extinguisher",
        "ball",
        "hat",
        "shower curtain rod",
        "water cooler",
        "paper cutter",
        "tray",
        "shower door",
        "pillar",
        "ledge",
        "toaster oven",
        "mouse",
        "toilet seat cover dispenser",
        "furniture",
        "cart",
        "storage container",
        "scale",
        "tissue box",
        "light switch",
        "crate",
        "power outlet",
        "decoration",
        "sign",
        "projector",
        "closet door",
        "vacuum cleaner",
        "candle",
        "plunger",
        "stuffed animal",
        "headphones",
        "dish rack",
        "broom",
        "guitar case",
        "range hood",
        "dustpan",
        "hair dryer",
        "water bottle",
        "handicap bar",
        "purse",
        "vent",
        "shower floor",
        "water pitcher",
        "mailbox",
        "bowl",
        "paper bag",
        "alarm clock",
        "music stand",
        "projector screen",
        "divider",
        "laundry detergent",
        "bathroom counter",
        "object",
        "bathroom vanity",
        "closet wall",
        "laundry hamper",
        "bathroom stall door",
        "ceiling light",
        "trash bin",
        "dumbbell",
        "stair rail",
        "tube",
        "bathroom cabinet",
        "cd case",
        "closet rod",
        "coffee kettle",
        "structure",
        "shower head",
        "keyboard piano",
        "case of water bottles",
        "coat rack",
        "storage organizer",
        "folded chair",
        "fire alarm",
        "power strip",
        "calendar",
        "poster",
        "potted plant",
        "luggage",
        "mattress",
    )

    HEAD_CATS_SCANNET_200 = [
        "tv stand",
        "curtain",
        "blinds",
        "shower curtain",
        "bookshelf",
        "tv",
        "kitchen cabinet",
        "pillow",
        "lamp",
        "dresser",
        "monitor",
        "object",
        "ceiling",
        "board",
        "stove",
        "closet wall",
        "couch",
        "office chair",
        "kitchen counter",
        "shower",
        "closet",
        "doorframe",
        "sofa chair",
        "mailbox",
        "nightstand",
        "washing machine",
        "picture",
        "book",
        "sink",
        "recycling bin",
        "table",
        "backpack",
        "shower wall",
        "toilet",
        "copier",
        "counter",
        "stool",
        "refrigerator",
        "window",
        "file cabinet",
        "chair",
        "wall",
        "plant",
        "coffee table",
        "stairs",
        "armchair",
        "cabinet",
        "bathroom vanity",
        "bathroom stall",
        "mirror",
        "blackboard",
        "trash can",
        "stair rail",
        "box",
        "towel",
        "door",
        "clothes",
        "whiteboard",
        "bed",
        "floor",
        "bathtub",
        "desk",
        "wardrobe",
        "clothes dryer",
        "radiator",
        "shelf",
    ]
    COMMON_CATS_SCANNET_200 = [
        "cushion",
        "end table",
        "dining table",
        "keyboard",
        "bag",
        "toilet paper",
        "printer",
        "blanket",
        "microwave",
        "shoe",
        "computer tower",
        "bottle",
        "bin",
        "ottoman",
        "bench",
        "basket",
        "fan",
        "laptop",
        "person",
        "paper towel dispenser",
        "oven",
        "rack",
        "piano",
        "suitcase",
        "rail",
        "container",
        "telephone",
        "stand",
        "light",
        "laundry basket",
        "pipe",
        "seat",
        "column",
        "bicycle",
        "ladder",
        "jacket",
        "storage bin",
        "coffee maker",
        "dishwasher",
        "machine",
        "mat",
        "windowsill",
        "bulletin board",
        "fireplace",
        "mini fridge",
        "water cooler",
        "shower door",
        "pillar",
        "ledge",
        "furniture",
        "cart",
        "decoration",
        "closet door",
        "vacuum cleaner",
        "dish rack",
        "range hood",
        "projector screen",
        "divider",
        "bathroom counter",
        "laundry hamper",
        "bathroom stall door",
        "ceiling light",
        "trash bin",
        "bathroom cabinet",
        "structure",
        "storage organizer",
        "potted plant",
        "mattress",
    ]
    TAIL_CATS_SCANNET_200 = [
        "paper",
        "plate",
        "soap dispenser",
        "bucket",
        "clock",
        "guitar",
        "toilet paper holder",
        "speaker",
        "cup",
        "paper towel roll",
        "bar",
        "toaster",
        "ironing board",
        "soap dish",
        "toilet paper dispenser",
        "fire extinguisher",
        "ball",
        "hat",
        "shower curtain rod",
        "paper cutter",
        "tray",
        "toaster oven",
        "mouse",
        "toilet seat cover dispenser",
        "storage container",
        "scale",
        "tissue box",
        "light switch",
        "crate",
        "power outlet",
        "sign",
        "projector",
        "candle",
        "plunger",
        "stuffed animal",
        "headphones",
        "broom",
        "guitar case",
        "dustpan",
        "hair dryer",
        "water bottle",
        "handicap bar",
        "purse",
        "vent",
        "shower floor",
        "water pitcher",
        "bowl",
        "paper bag",
        "alarm clock",
        "music stand",
        "laundry detergent",
        "dumbbell",
        "tube",
        "cd case",
        "closet rod",
        "coffee kettle",
        "shower head",
        "keyboard piano",
        "case of water bottles",
        "coat rack",
        "folded chair",
        "fire alarm",
        "power strip",
        "calendar",
        "poster",
        "luggage",
    ]
    VALID_CLASS_IDS_200_VALIDATION = (
        "wall",
        "chair",
        "floor",
        "table",
        "door",
        "couch",
        "cabinet",
        "shelf",
        "desk",
        "office chair",
        "bed",
        "pillow",
        "sink",
        "picture",
        "window",
        "toilet",
        "bookshelf",
        "monitor",
        "curtain",
        "book",
        "armchair",
        "coffee table",
        "box",
        "refrigerator",
        "lamp",
        "kitchen cabinet",
        "towel",
        "clothes",
        "tv",
        "nightstand",
        "counter",
        "dresser",
        "stool",
        "cushion",
        "plant",
        "ceiling",
        "bathtub",
        "end table",
        "dining table",
        "keyboard",
        "bag",
        "backpack",
        "toilet paper",
        "printer",
        "tv stand",
        "whiteboard",
        "blanket",
        "shower curtain",
        "trash can",
        "closet",
        "stairs",
        "microwave",
        "stove",
        "shoe",
        "computer tower",
        "bottle",
        "bin",
        "ottoman",
        "bench",
        "board",
        "washing machine",
        "mirror",
        "copier",
        "basket",
        "sofa chair",
        "file cabinet",
        "fan",
        "laptop",
        "shower",
        "paper",
        "person",
        "paper towel dispenser",
        "oven",
        "blinds",
        "rack",
        "plate",
        "blackboard",
        "piano",
        "suitcase",
        "rail",
        "radiator",
        "recycling bin",
        "container",
        "wardrobe",
        "soap dispenser",
        "telephone",
        "bucket",
        "clock",
        "stand",
        "light",
        "laundry basket",
        "pipe",
        "clothes dryer",
        "guitar",
        "toilet paper holder",
        "seat",
        "speaker",
        "column",
        "ladder",
        "bathroom stall",
        "shower wall",
        "cup",
        "jacket",
        "storage bin",
        "coffee maker",
        "dishwasher",
        "paper towel roll",
        "machine",
        "mat",
        "windowsill",
        "bar",
        "toaster",
        "bulletin board",
        "ironing board",
        "fireplace",
        "soap dish",
        "kitchen counter",
        "doorframe",
        "toilet paper dispenser",
        "mini fridge",
        "fire extinguisher",
        "ball",
        "hat",
        "shower curtain rod",
        "water cooler",
        "paper cutter",
        "tray",
        "shower door",
        "pillar",
        "ledge",
        "toaster oven",
        "mouse",
        "toilet seat cover dispenser",
        "furniture",
        "cart",
        "scale",
        "tissue box",
        "light switch",
        "crate",
        "power outlet",
        "decoration",
        "sign",
        "projector",
        "closet door",
        "vacuum cleaner",
        "plunger",
        "stuffed animal",
        "headphones",
        "dish rack",
        "broom",
        "range hood",
        "dustpan",
        "hair dryer",
        "water bottle",
        "handicap bar",
        "vent",
        "shower floor",
        "water pitcher",
        "mailbox",
        "bowl",
        "paper bag",
        "projector screen",
        "divider",
        "laundry detergent",
        "bathroom counter",
        "object",
        "bathroom vanity",
        "closet wall",
        "laundry hamper",
        "bathroom stall door",
        "ceiling light",
        "trash bin",
        "dumbbell",
        "stair rail",
        "tube",
        "bathroom cabinet",
        "closet rod",
        "coffee kettle",
        "shower head",
        "keyboard piano",
        "case of water bottles",
        "coat rack",
        "folded chair",
        "fire alarm",
        "power strip",
        "calendar",
        "poster",
        "potted plant",
        "mattress",
    )

    def load(self, filename):
        scan_id = osp.basename(filename).replace(self.suffix, "")

        if self.prefix == "test":
            xyz, rgb = torch.load(filename)
            semantic_label = np.zeros(xyz.shape[0], dtype=np.long)
            instance_label = np.zeros(xyz.shape[0], dtype=np.long)
        else:
            xyz, rgb, semantic_label, instance_label = torch.load(filename)

        spp_filename = osp.join(self.data_root, "superpoints", scan_id + ".pth")
        spp = torch.load(spp_filename)

        instance_label[semantic_label <= 1] = -100
        return xyz, rgb, semantic_label, instance_label, spp