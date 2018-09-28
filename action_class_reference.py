import os
import json

super_actions = os.listdir("../EgoK360_Data")

actions = []
sub_actions = []

for super_action in sorted(super_actions):
    print("Super: ")
    print(super_action)
    sub_actions = os.listdir("../EgoK360_Data" + "/" + super_action)
    for sub_action in sorted(sub_actions):
        if sub_action not in actions:
            actions.append(sub_action)

actions = sorted(actions)
print(actions)
print(len(actions))

action_dictionary = {}

for i, action in enumerate(actions):
    action_dictionary[action] = i

print(action_dictionary)

json.dump(action_dictionary, open("actions.txt", 'w'))
