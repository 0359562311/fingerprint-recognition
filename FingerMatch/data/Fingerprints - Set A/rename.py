import os

count = 0

for i in os.listdir():
    if i != "rename.py":
        count = count + 1
        os.rename(i, str(count) + ".tif")
        