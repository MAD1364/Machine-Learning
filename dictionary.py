import json


partition = json.load(open("partition_dict.txt"))
print(partition['validation'])

labels = json.load(open("labels_dict.txt"))
print(labels)
