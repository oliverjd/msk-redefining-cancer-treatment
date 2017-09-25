def read_data():
	train = list()
	with open('data/training_text', 'r') as file:
		next(file)
		for line in file:
			line = line.rstrip('\n')
			fields = line.split('||')
			train.append({'text': fields[1]})
	with open('data/training_variants', 'r') as file:
		next(file)
		i=0
		for line in file:
			line = line.rstrip('\n')
			fields = line.split(',')
			train[i]['gene'] = fields[1]
			train[i]['variation'] = fields[2]
			train[i]['class'] = fields[3]
			i += 1
	test = list()
	with open('data/test_text', 'r') as file:
		next(file)
		for line in file:
			line = line.rstrip('\n')
			fields = line.split('||')
			test.append({'text': fields[1]})
	with open('data/test_variants', 'r') as file:
		next(file)
		i=0
		for line in file:
			line = line.rstrip('\n')
			fields = line.split(',')
			test[i]['gene'] = fields[1]
			test[i]['variation'] = fields[2]
			i += 1
	return (train, test)

(train, test) = read_data()

for i in range(len(train)):
	filename = 'train_data/' + str(train[i]['class']) + '/' + str(i)
	f = open(filename, 'w')
	f.write(train[i]['text'])
	f.close()

for i in range(len(test)):
	filename = 'test_data/' + str(i)
	f = open(filename, 'w')
	f.write(test[i]['text'])
	f.close()