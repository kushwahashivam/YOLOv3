img_size = 640

lbl_fp = open("labels.txt", "r")
cats = [line for line in lbl_fp]
lbl_fp.close()
coco_cat2int = dict()
for c in cats:
    i, c = c.split(":")
    coco_cat2int[c.strip()] = int(i)
coco_int2cat = dict(zip(coco_cat2int.values(), coco_cat2int.keys()))

cat2cat = {
    "person": "person", 
    "bicycle": "vehicle", 
    "boat": "vehicle", 
    "bus": "vehicle", 
    "car": "vehicle", 
    "motorcycle": "vehicle", 
    "truck": "vehicle", 
    "cat": "animal", 
    "cow": "animal", 
    "dog": "animal", 
    "elephant": "animal", 
    "horse": "animal", 
    "bird": "bird"
}

cat2int = {
    "person" : 0,
    "vehicle": 1, 
    "animal": 2, 
    "bird": 3
}

int2cat = dict(zip(cat2int.values(), cat2int.keys()))