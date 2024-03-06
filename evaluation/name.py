CLASSES_200 = (
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


CLASSES_20 = (
        "cabinet",
        "bed",
        "chair",
        "sofa",
        "table",
        "door",
        "window",
        "bookshelf",
        "picture",
        "counter",
        "desk",
        "curtain",
        "refrigerator",
        "shower curtain",
        "toilet",
        "sink",
        "bathtub",
        "otherfurniture",
    )


# BASE_CLASSES = ['cabinet', 'kitchen cabinet', 'file cabinet', 'bathroom cabinet', 'bed',\
#             'chair', 'office chair', 'armchair', 'sofa chair', 'folded chair', 'sofa chair',\
#             'table', 'coffee table', 'end table', 'dining table', 'door', 'doorframe', \
#             'shower door', 'closet door', 'bathroom stall door', 'window', 'windowsill', \
#             'bookshelf', 'picture', 'counter', 'kitchen counter', 'bathroom counter', \
#             'desk', 'curtain', 'shower curtain', 'shower curtain rod', 'refrigerator', \
#             'shower curtain', 'shower curtain rod', 'toilet', 'toilet paper', 'toilet paper holder',\
#             'toilet paper dispenser', 'toilet seat cover dispenser', 'sink', 'bathtub']
NOVEL_CLASSES = []


# for cls in CLASSES_20:
#     for new_cls in CLASSES_200:
#         if cls in new_cls:
#             BASE_CLASSES.append(new_cls)


# for cls in CLASSES_200:
#     if cls not in BASE_CLASSES:
#         NOVEL_CLASSES.append(cls)

# print(NOVEL_CLASSES)

# print('\n\nbase')
# print(BASE_CLASSES, len(BASE_CLASSES))

BASE_CLASSES = []
with open('base_classes.txt') as f:
    lines = []
    for line in f:
        BASE_CLASSES.append(line.strip())

for cls in CLASSES_200:
    if cls not in BASE_CLASSES:
        NOVEL_CLASSES.append(cls)

print(NOVEL_CLASSES)
# print(BASE_CLASSES, len(BASE_CLASSES))


ind = range(2,20)

for i in ind:
    b_cls = CLASSES_20[i-2]

    for i, c in enumerate(CLASSES_200):
        if c == b_cls:
            print(b_cls, i+2)

# for cls in ["sofa chair"]:
#     for i, c in enumerate(CLASSES_200):
#         if c == cls:
#             print(c, i+2)

