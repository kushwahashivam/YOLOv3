cat2class = {
    "person": "person", 
    "boat": "vehicle", 
    "bus": "vehicle", 
    "car": "vehicle", 
    "truck": "vehicle", 
    "cat": "animal", 
    "cow": "animal", 
    "dog": "animal", 
    "elephant": "animal", 
    "horse": "animal", 
    "bird": "bird"
}

class2int = {
    "person" : 0,
    "vehicle": 1, 
    "animal": 2, 
    "bird": 3
}

int2class = dict(zip(class2int.values(), class2int.keys()))