f = open(r"VGG16\with_augmentation\results.txt","r")

lines = f.readlines()
f.close()

lines = [line.strip() for line in lines if len(line) > 0 and line[0] == 'A']

lines = lines[1:]

for each in lines:
    metrics = each.split(',')
    print (round(float(metrics[4].split(':')[1]),5))