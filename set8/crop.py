def get_data_set(filename):
    file = open(filename)
    data = []
    for line in file:
        data.append(list(map(float, filter(bool, line.split()))))
    return data


def crop(filename):
	data = get_data_set(filename)
	file = open(filename + "_converted.txt", 'w')
	file.write(str(data))


#crop("features_test.txt")
crop("features_train.txt")