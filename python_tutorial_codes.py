#please use utf-8 to run these codes
a = 2
b = float(a)
c = b
b = 3.0
print(c)
small_birds = ['hummingbird', 'finch']
extinct_birds = ['dodo', 'passenger pigeon', 'norwegian blue']
carol_birds = [3, 'french hens', 2, 'turtle doves']
all_birds = [small_birds, extinct_birds, 'macaw', carol_birds]
extinct_birds[2] = 'diggie'
print(extinct_birds)
marxes = ('Groucho', 'Chico', 'Harpo','Zeppo', 'Gummo')
others = ['Gummo', 'Karl']
b = 'Chico' in marxes
print(marxes.index('Zeppo'))
a = sorted(marxes)
a.insert(2,'Karl')
print(a)
a = [1,2,3]
b = a.copy()
c = list(a)
d = a[:]
a[0] = 'integer lists are boring'
print(a)
print(b)
print(c)
print(d)
marx_tuple = 'Groucho','Chico', 'Harpo'
a,b,c = marx_tuple
print(marx_tuple)
print(a)
print(b)
print(c)
password = 'swordfish'
icecream = 'tuttifrutti'
password, icecream = icecream, password
print(password)
print(icecream)
marx_list = ['Groucho','Chico', 'Harpo']
print(tuple(marx_list))
#use {} to build dictionary
bierce = {
	"day": "A period of twenty-four hours, mostly misspent",
	"positive": "Mistaken at the top of one's voice",
	"misfortune": "The kind of fortune that never misses",
}
print(bierce)
#The dict() constructor builds dictionaries directly from sequences of key-value pairs:
lol = [['a', 'b'], ['c', 'd'], ['e', 'f']]
print(dict(lol))
lot = [('a', 'b'), ('c', 'd'), ('e', 'f')]
print(dict(lot))
tol = (['a', 'b'], ['c', 'd'], ['e', 'f'])
print(dict(tol))
los = ['ab', 'cd', 'ef']
print(dict(los))
tos = ('ab', 'cd', 'ef')
print(dict(los))
#the [key] adds or revises element
pythons = {
	'Chapman': 'Graham',
	'Cleese': 'John',
	'Idle': 'Eric',
	'Jones': 'Terry',
	'Palin': 'Michael'
}
pythons['Gilliam'] = 'Gerry'
pythons['Gilliam'] = 'Terry'
print(pythons) 
#Combine Dictionaries with update(). Delete an Item by Key with del. Delete All Items by Using clear() 
pythons = {
	'Chapman': 'Graham',
	'Cleese': 'John',
	'Idle': 'Eric',
	'Jones': 'Terry',
	'Palin': 'Michael',
	'Gilliam': 'Terry'
}
others = {
	'Marx': 'Groucho',
	'Howard': 'Moe'
}
pythons.update(others)
print(pythons)
del pythons['Marx']
print(pythons)
del pythons['Howard']
print(pythons)
pythons.clear()
print(pythons)
pythons = {
	'Chapman': 'Graham',
	'Cleese': 'John',
	'Idle': 'Eric',
	'Jones': 'Terry',
	'Palin': 'Michael',
	'Gilliam': 'Terry'
}
others = {
	'Marx': 'Groucho',
	'Howard': 'Moe'
}
pythons.update(others)
print('Chapman' in pythons)
print('Pulin' in pythons)
print(pythons['Marx'])
print(pythons.get('Marx','NOT A PYTHON'))

signals = {'green': 'go', 'yellow': 'go faster', 'red': 'smile for the camera'}
print(signals)
print(signals.keys())
print(list(signals.keys()))
print(list(signals.values()))
print(list(signals.items()))
save_signals = signals
original_signals = signals.copy()
save_signals['blue'] = 'conduse everyone'
print(save_signals)
print(original_signals)

empty_set = set()
print(empty_set)
even_numbers = {0, 2, 4, 6, 8}
print(even_numbers)
odd_numbers = {1, 3, 5, 7, 9}
print(odd_numbers)
print(set('letters'))
print(set(['Dasher', 'Dancer', 'Prancer', 'Mason-dixon']))
print(set(('Ummagumma', 'Echoes', 'Atom Heart Mother')))
print(set({'apple': 'red', 'orange': 'orange', 'cherry': 'red'}))

drinks= {
	'martini': {'vodka', 'vermouth'},
	'black russian': {'vodka', 'kahlua'},
	'white russian': {'cream', 'kahlua', 'vodka'},
	'manhattan': {'rye', 'vermouth', 'bitters'},
	'screwdriver': {'orange juice', 'vodka'}
}
print(drinks.items())
print('----------------------------')
for name, contents in drinks.items():
	if 'vodka' in contents:
		print(name)
print('----------------------------')
for name, contents in drinks.items():
	if 'vodka' in contents and not ('vermouth' in contents or 'cream' in contents):
		print(name)
print('----------------------------')
for name, contents in drinks.items():
	if contents & {'vermouth', 'orange juice'}:
		print(name)
print('----------------------------')
for name, contents in drinks.items():
	if 'vodka' in contents and not contents & {'vermouth', 'cream'}:
		print(name)
print('----------------------------')
bruss = drinks['black russian']
wruss = drinks['white russian']
a = {1, 2}
b = {2,3}
print(a & b)
print('----------------------------')
print(a.intersection(b))
print('----------------------------')
print(bruss & wruss)
print('----------------------------')
print(a | b)
print('----------------------------')
print(a.union(b))
print('----------------------------')
print(bruss | wruss)
print('----------------------------')
print(a - b)
print('----------------------------')
print(a.difference(b))
print('----------------------------')
print(bruss - wruss)
print('----------------------------')
print(bruss.difference(wruss))
print('----------------------------')
print(wruss - bruss)
print('----------------------------')
print(wruss.difference(bruss))
print('----------------------------')
print(a ^ b)
print('----------------------------')
print(a.symmetric_difference(b))
print('----------------------------')
print(bruss ^ wruss)
print('----------------------------')
print(a <= b)
print('----------------------------')
print(a.issubset(b))
print('----------------------------')
print(bruss <= wruss)
print('----------------------------')
print(a < a)
print('----------------------------')
print(bruss < wruss)
print('----------------------------')
print(wruss > bruss)
print('----------------------------')

marx_list = ['Groucho', 'Chico', 'Harpo']
marx_tuple = 'Groucho', 'Chico', 'Harpo'
marx_dict = {'Groucho': 'banjo',  'Chico': 'piano', 'Harpo': 'harp'}
print(marx_list[2])
print(marx_tuple[2])
print(marx_dict['Harpo'])
house = {
	(44.79, -93.14, 285): 'My House',
	(38.89, -77.03, 13): 'The White House'
}
print(house)
print('----------------------------')

alphabet = 'abcdefg'
alphabet += 'hijklmn'
print(alphabet)
disaster = True
if disaster:
	print("Woe!")
else:
	print("Whee!")
print("------------------")
furry = True
small = True
if furry:
	if small:
		print("It's a cat.")
	else:
		print("It's a bear!")
else:
	if small:
		print("It's a skink")
	else:
		print("It's a human. Or a hairless bear.")
print("------------------")
color = "puce"
if color == "red":
	print("It's is a tomato")
elif color == "green":
	print("It's a green pepper")
elif color == "bee purple":
	print("I dont know what it is, but only bees can see it")
else:
	print("I've never heard of the color", color)
print("------------------")
some_list = []
if some_list:
	print("There's something in here!")
else:print("Hey, it's empty!")
print("------------------")
count = 1
while count <=5:
	print(count)
	count += 1
print("------------------")
while True:
	stuff = input("String to capitalize [type exit to quit]: ")
	if stuff == 'exit':
		break
	print(stuff.capitalize())
print("------------------")
while True:
	value = input("Integer, please [q to quit]: ")
	if value == 'q':
		break
	number = int(value)
	if number % 2 == 0:
		continue
	print(number, 'squared is', number*number)
numbers = [1, 3, 5]
position = 0
while position < len(numbers):
	number = numbers[position]
	if number % 2 == 0:
		print('Found even number', number)
		break
	position += 1
else:
	print('No even found')
rabbits = ['Flopsy', 'Mopsy', 'Cottontail', 'Peter']
current = 0
while current < len(rabbits):
	print(rabbits[current])
	current += 1
print("------------------")
for rabbit in rabbits:
	print(rabbit)
print("------------------")
word = 'cats'
for letter in word:
	print(letter)
print("------------------")
accusation = {'room': 'ballroom', 'weapon': 'lead pipe', 'person': 'Col.Mustard'}
for card in accusation:
	print(card)
print("------------------")
for card in accusation.keys():
	print(card)
print("------------------")
for card in accusation.values():
	print(card)
print("------------------")
for card in accusation.items():
	print(card)
print("------------------")
for card, contents in accusation.items():
	print('Card', card, 'has the contents', contents)
print("------------------")
cheeses = []
for cheese in cheeses:
	print('This shop has some lovely', cheese)
	break
else:
	print('This is not much of a cheese shop, is it?')
print("------------------")
days = ['Monday', 'Tuesday', 'Wedensday']
fruits = ['banana', 'orange', 'peach']
drinks = ['coffee', 'tea', 'beer']
desserts = ['tiramisu', 'ice cream', 'pie', 'pudding']
for day, fruit, drink, dessert in zip(days, fruits, drinks, desserts):
	print(day, ": drink", drink, "- eat", fruit, "- enjoy", dessert)
english = 'Monday', 'Tuesday', 'Wedensday'
french = 'Lundi', 'Mardi', 'Mercredi'
print(list(zip(english, french)))
print(dict(zip(english, french)))
for x in range(0, 100, 4):
	print(x)
print(list(range(0, 100, 4)))
for x in range(100, -100, -2):
	print(x)
print(list(range(100, -100, -2)))
number_list = []
i = 1
while i <= 5:
	number_list.append(i)
	i += 1
print(number_list)
odd_list = []
for i in range(1, 10, 2):
	odd_list.append(i)
print(odd_list)
number_list = list(range(1, 10, 2))
print(number_list)
number_list = [number for number in range(1,6)]
print(number_list)
number_list = [number - 1 for number in range(1,6)]
print(number_list)
a_list = [number * 2 + 1 for number in range(1, 6) if number % 2 == 1]
print(a_list)
a_list = []
for number in range(1,6):
	if number % 2 == 1:
		a_list.append(number)
print(a_list)
rows = range(1,40,1)
cols = range(1,30,1)
for row in rows:
	for col in cols:
		print(row, col)
cells = [(row, col) for row in rows if row % 4 == 1 for col in cols if col % 3 ==2]
for cell in cells:
	print(cell)
for row, col in cells:
	print(row, col)
word = 'letters'
letter_counts = {letter: word.count(letter) for letter in word}
print(letter_counts)
letter_counts = {letter: word.count(letter) for letter in set(word)}
print(letter_counts)
a_set = {number for number in range(1,100) if number % 3 == 1}
print(a_set)
number_thing = (number for number in range(1,6))
for number in number_thing:
	print(number)
number_thing = (number for number in range(1,6))
number_list = list(number_thing)
print(number_list)
number_thing = (number for number in range(1,6))
number_list = list(number_thing)
print(number_list)
def make_a_sound():
	print('quack')
make_a_sound()
def agree():
	return True
if agree():
	print('Splendid!')
else:
	print('That was unexpected.')
def echo(anything):
	print('This is an echo function!')
	return anything + ' ' + anything
print(echo('hello'))
def commentary(color):
	if color == 'red':
		return "It's a tomato."
	elif color == 'green':
		return "It's a green pepper."
	elif color == 'bee purple':
		return "I don't know what it is, but only bees can see it."
	else:
		return "I've never heard of the color " + color + "."
while True:
	value = input("Input a color('q' to exit):")
	if value == 'q':
		break
	else:
		print(commentary(value))
input('press anything to quit...')
def meau(wine, entree, dessert):
	return {'wine': wine, 'entree': entree, 'dessert': dessert}
print(meau('chardonnay', 'chicken', 'cake'))
print(meau(entree = 'beef', dessert = 'bagel', wine = 'bordeaux'))
def buggy(arg, result = []):
	result.append(arg)
	return result
print(buggy('a'))
print(buggy('b'))
def works(arg):
	result = []
	result.append(arg)
	return result
print(works('a'))
print(works('b'))
def non(arg, result = None):
	if result is None:
		result = []
	result.append(arg)
	print(result)
non('a')
non('b')
def print_args(*args):
	print('Position argument tuple: ', args)
print_args()
print_args(3, 2, 1, 'wait!', 'uh...')
def  print_more(required1, required2, *args):
	print('Need this one: ', required1)
	print('Need this one too: ', required2)
	print('All the rest: ', args)
print_more('cap', 'gloves', 'scarf', 'monocle', 'mustache waxasterisk')
def print_kwargs(**print_kwargs):
	print('Keyword arguments: ', print_kwargs)
print_kwargs(wine = 'merlot', entree = 'mutton', dessert = 'macaroon')
def echo(anything):
	'echo returns its imput argument'
	return anything
help(echo)
print('---------------------')
print(echo.__doc__)
def add_args(arg1, arg2):
	print(arg1 + arg2)
print(type(add_args(1, 2)))
def knights(saying):
	def inner(quote):
		return "We are the knights who says: '%s'" %quote
	return inner(saying)
print(knights('Ni!'))
def knights2(saying):
	def inner2():
		return "We are the knights who say: '%s'" % saying
	return inner2
print(type(knights2('Duck')))
print(type(knights2('Hasenpfeffer')))
print(knights2('Duck'))
print(knights2('Hasenpfeffer'))
print(knights2('Duck')())
print(knights2('Hasenpfeffer')())
def edit_story(words, func):
	for word in words:
		print(func(word))
stairs = ['thub', 'meow', 'thub', 'hiss']
def enliven(word):
	return word.capitalize() + '!'
print(edit_story(stairs, enliven))
def edit_story(words, func):
	for word in words:
		print(func(word))
stairs = ['thub', 'meow', 'thub', 'hiss']
print(edit_story(stairs, lambda word: word.capitalize() + '!'))
def my_range(first = 0, last = 10, step = 1):
	number = first
	while number < last:
		yield number
		number += step
print(my_range)
ranger = my_range(1, 5)
print(ranger)
for x in ranger:
	print(x)
print('---------------')
def document_it(func):
	def new_function(*args, **kwargs):
		print('Running function: ', func.__name__)
		print('Positional arguments: ', args)
		print('Keyword argument: ', kwargs)
		result = func(*args, **kwargs)
		print('Result: ', result)
		return result
	return new_function
def add_ints(a, b):
	return a + b
cooler_add_ints = document_it(add_ints)
print(cooler_add_ints(3, 5))
def square_it(func):
	def new_function(*args, **kwargs):
		result = func(*args, **kwargs)
		return result * result
	return new_function
def document_it(func):
	def new_function(*args, **kwargs):
		print('Running function: ', func.__name__)
		print('Positional arguments: ', args)
		print('Keyword argument: ', kwargs)
		result = func(*args, **kwargs)
		print('Result: ', result)
		return result
	return new_function
@square_it
@document_it
def add_ints(a, b):
	return a + b
a = add_ints(3, 5)
print(a)
animal = 'fruitbat'
print(id(animal))
def print_global():
	print('inside print_global:', animal)
print('at the top level:', animal)
print_global()
def change_local():
	global animal
	animal = 'wombat'
	print('inside change_local:', animal, id(animal))
change_local()
print(id(animal))
animal = 'fruitbat'
def change_local():
	animal = 'wombat'
	print('locals:', locals())
print(animal)
change_local()
print('globals:', globals())
short_list = [1, 2, 3]
while True:
	value =input('Position [q to quit]? ')
	if value == 'q':
		break
	try:
		position = int(value)
		print(short_list[position])
	except IndexError as  err:
		print('Bad index:', position)
	except Exception as other:
		print('Something else broke:', other)
# the following codes should be run in a new module!!!
#class UppercaseException(Exception):
#	pass
#words = ['eeenie', 'meenie', 'miny', 'MO']
#for word in words:
#	if word.isupper():
#		raise UppercaseException(word)
input('Press any key to continue...')
while True:
	guess_me = 7
	a = input('input a number:')
	if a.isdigit() is False:
		print('Error, input a number!')
	elif int(a) == 7:
		print('just right')
		break
	elif int(a) > 7:
		print('too high')
	else:
		print('too low')
input('Press any key to continute...')
a = [3, 2, 1, 0]
for b in a:
	print(b)
a = [ word for word in range(0, 10, 1) if word % 2 == 0 ]
print(a)
squares = {word: word * word for word in range(10)}
print(squares)
odd = {a for a in range(10) if a % 2 == 1}
print(odd)
for thing in ('Got %s' % num for num in range(10)):
	print(thing)
def good():
	a_list = ['Harry', 'Ron', 'Hermione']
	return a_list
print(good())
def get_odds():
	a = 0
	while a < 10:
		if a %2 == 1:
			yield a
		a += 1
i = 1
for word in get_odds():
	if i == 3:
		print(word)
	i += 1
def document_it(func):
	def new_function(args, kwargs):
		print('Start')
		result = func(args, kwargs)
		print('End')
	return new_function
@document_it
def add_func(a, b):
	print(a, b, a + b)
print(add_func(2, 5))
titles = ['creature of habit', 'crewel fate']
plots = ['a nun turns into a monster', 'a haunted yarn shop']
movies = {title: plot for title, plot in zip(titles, plots)}
print(movies)
import sys
for place in sys.path:
     print(place)
periodic_table = {'Hydrogen': 1, 'Helium': 2}
print(periodic_table)
carbon = periodic_table.setdefault('Carbon', 12)
print(carbon)
print(periodic_table)
helium = periodic_table.setdefault('Helium', 947)
print(helium)
print(periodic_table)
from collections import defaultdict
periodic_table = defaultdict(int)
periodic_table['Hydrogen'] = 1
print(periodic_table['Lead'])
print(periodic_table)
from collections import defaultdict
def no_idea():
	return 'Huh?'
bestiary = defaultdict(no_idea)
bestiary['A'] = 'Abominable Snowman'
bestiary['B'] = 'Basilisk'
print(bestiary)
print(bestiary['C'])
print(bestiary)
from collections import defaultdict
bestiary = defaultdict(lambda: 'Huh?')
print(bestiary['E'])
from collections import defaultdict
food_counter = defaultdict(int)
for food in ['spam', 'spam', 'eggs', 'spam']:
	food_counter[food] += 1
for food, count in food_counter.items():
	print(food, count)
	dict_counter = {}
for food in ['spam', 'spam', 'eggs', 'spam']:
	if not food in dict_counter:
		dict_counter[food] = 0
	dict_counter[food] += 1
for food, count in dict_counter.items():
	print(food, count)
from collections import Counter
breakfast = ['spam', 'spam', 'eggs', 'spam']
breakfast_counter = Counter(breakfast)
print(breakfast_counter)
from collections import Counter
breakfast = ['spam', 'spam', 'eggs', 'spam']
breakfast_counter = Counter(breakfast)
print(breakfast_counter)
print(breakfast_counter.most_common())
print(breakfast_counter.most_common(1))
print(breakfast_counter.most_common(2))
from collections import Counter
breakfast = ['spam', 'spam', 'eggs', 'spam']
breakfast_counter = Counter(breakfast)
print(breakfast_counter)
print(breakfast_counter.most_common())
print(breakfast_counter.most_common(1))
print(breakfast_counter.most_common(2))
print('----------------------------')
lunch = ['eggs', 'eggs', 'bacon']
lunch_counter = Counter(lunch)
print(lunch_counter)
print('----------------------------')
print(breakfast_counter + lunch_counter)
print('----------------------------')
print(breakfast_counter - lunch_counter)
print('----------------------------')
print(lunch_counter - breakfast_counter)
print('----------------------------')
print(breakfast_counter & lunch_counter)
print('----------------------------')
print(breakfast_counter | lunch_counter)
print('----------------------------')
quotes = {
	'Moe': 'A wise guy, huh?',
	'Larry': 'Ow!',
	'Curly': 'Nyuk nyuk!'
}
for stooge in quotes:
	print(stooge)
print('----------------------------')
from collections import OrderedDict
quotes = OrderedDict([
	('Moe', 'A wise guy, huh'),
	('Larry', 'Ow!'),
	('Curly', 'Nyuk nyuk!')
])
for stooge in quotes:
	print(stooge)
def palindrome(word):
	from collections import deque
	dq = deque(word)
	while len(dq) > 1:
		if dq.popleft() != dq.pop():
			return False
	return True
print(palindrome('a'))
print(palindrome('racecar'))
print(palindrome(''))
print(palindrome('halibut'))
def another_palindrome(word):
	return word == word[::-1]
print(another_palindrome('radar'))
print(another_palindrome('rookie'))
import itertools
for item in itertools.chain([1, 2], ['a', 'b']):
	print(item)
for item in itertools.accumulate([1, 2, 3, 4], lambda a, b: a * b):
	print(item)
from collections import OrderedDict
from pprint import pprint
quotes = OrderedDict([
	('MOe', 'A wise guy, huh?'),
	('Larry', 'Ow!'),
	('Curly', 'Nyuk nyuk!')
	])
print(quotes)
pprint(quotes)
plain = {
	'a': 1,
	'b': 2,
	'c': 3
}
print(plain)
from collections import OrderedDict
fancy = OrderedDict([
	('a', 1),
	('b', 2),
	('c', 3)
	])
print(fancy)
class Person():
	def __init__(self, name):
		self.name = name
hunter = Person('Elmer Fudd')
print('The mighty hunter:', hunter.name)
#2016-8-19
class Car():
	def exclaim(self):
		print("I'm a Car!")
class Yugo(Car):
	pass
give_me_a_car = Car()
giive_me_a_yugo = Yugo()
give_me_a_car.exclaim()
giive_me_a_yugo.exclaim()
car = Car()
Car.exclaim(car)
class Car():
	def exclaim(self):
		print("I'm a Car!")
class Yugo(Car):
	"""docstring for Yugo"""
	def exclaim(self):
		print("I'm a Yugo! Much like a Car, but more Yugo-ish.")
give_me_a_car = Car()
give_me_a_yugo = Yugo()
give_me_a_car.exclaim()
give_me_a_yugo.exclaim()
class Person():
	"""docstring for Person"""
	def __init__(self, name):
		self.name = name
class MDPerson(Person):
	"""docstring for MDPerson"""
	def __init__(self, name):
		self.name = "Doctor, " + name
class JDPerson(Person):
			"""docstring for JDPerson"""
			def __init__(self, name):
				self.name = name + ", Esquire"
person = Person('Fudd')
doctor = MDPerson('Fudd')
lawyer = JDPerson('Fudd')	
print(person.name)
print(doctor.name)
print(lawyer.name)
class Car():
	def exclaim(self):
		print("I'm a Car!")
class Yugo(Car):
	"""docstring for Yugo"""
	def exclaim(self):
		print("I'm a Yugo! Much like a Car, but more Yugo-ish.")
	def need_a_push(self):
		print("A little help here?")
give_me_a_car = Car()
give_me_a_yugo = Yugo()
give_me_a_yugo.need_a_push()
class Person():
	def __init__(self, name):
		self.name = name
class EmailPerson(Person):
	def __init__(self, name, email):
		super().__init__(name)
		self.email = email
bob = EmailPerson('Bob Frapples', 'bob@frapples.com')
print(bob.name)
print(bob.email)
class Person():
	def __init__(self, name):
		self.name = name
class EmailPerson(Person):
	def __init__(self, name, email):
		super(EmailPerson, self).__init__(name)
		self.email = email
bob = EmailPerson('Bob Frapples', 'bob@frapples.com')
print(bob.name)
print(bob.email)
#2016-8-20
class Duck():
	def __init__(self, input_name):
		self.hidden_name = input_name
	def get_name(self):
		print('inside the getter')
		return self.hidden_name
	def set_name(self, input_name):
		print('inside the setter')
		self.hidden_name = input_name
	name = property(get_name, set_name)
fowl = Duck('Howard')
print(fowl.name)
print('----------------------------')
print(fowl.get_name())
print('----------------------------')
fowl.name = 'Daffy'
print('----------------------------')
print(fowl.name)
print('----------------------------')
fowl.set_name('Daffy')
print('----------------------------')
print(fowl.name)
print('----------------------------')
print(fowl.hidden_name)
class Duck():
	def __init__(self, input_name):
		self.hidden_name = input_name
	@property
	def name(self):
		print('inside the getter')
		return self.hidden_name
	@name.setter
	def name(self, input_name):
		print('inside the setter')
		self.hidden_name = input_name
print('----------------------------')
fowl = Duck('Howard')
print(fowl.name)
print('----------------------------')
fowl.name = 'Donald'
print('----------------------------')
print(fowl.name)
print('----------------------------')
class Circle():
	def __init__(self, radius):
		self.radius = radius
	@property
	def diameter(self):
		return 2 * self.radius
c = Circle(5)
print(c.radius)
print('----------------------------')
print(c.diameter)
print('----------------------------')
c.radius = 7
print(c.diameter)
print('----------------------------')
class Duck():
	def __init__(self, input_name):
		self.__name = input_name
	@property
	def name(self):
		print('inside the getter')
		return self.__name
	@name.setter
	def name(self, input_name):
		print('inside the setter')
		self.__name = input_name
fowl = Duck('Howard')
print(fowl.name)
print('----------------------------')
fowl.name = 'Donald'
print('----------------------------')
print(fowl.name)
print('----------------------------')
print(fowl._Duck__name)
print('----------------------------')
class A():
	count = 0
	def __init__(self):
		A.count += 1
	def exclaim(self):
		print("I'm an A!")
	@classmethod
	def kids(cls):
		print("A has", cls.count, 'little objects.')
easy_a = A()
breezy_a = A()
wheezy_a = A()
A.kids()
class CoyoteWeapon():
	@staticmethod
	def commercial():
		print('This Coyoteweapon has been brought to you by Acme')
CoyoteWeapon.commercial()
print('----------------------------')
class Quote():
	def __init__(self, person, words):
		self.person = person
		self.words = words
	def who(self):
		return self.person
	def says(self):
		return self.words + '.'
class QuestionQuote(Quote):
	"""docstring for QuestionQuote"""
	def says(self):
		return self.words + '?'
class ExclaimationQuote(Quote):
	"""docstring for ExclaimationQuoye"""
	def says(self):
		return self.words + '!'
hunter = Quote('Elmer Fudd', "I'm hunting wabbits")
print(hunter.who(), 'says:', hunter.says())
hunted1 = QuestionQuote('Bugs Bunny', "What's up, doc")
print(hunted1.who(), 'says:', hunted1.says())
hunted2 = ExclaimationQuote('Daffy Duck', "It's rabbit season")
print(hunted2.who(), 'says:', hunted2.says())
print('----------------------------')
class Quote():
	def __init__(self, person, words):
		self.person = person
		self.words = words
	def who(self):
		return self.person
	def says(self):
		return self.words + '.'
class QuestionQuote(Quote):
	"""docstring for QuestionQuote"""
	def says(self):
		return self.words + '?'
class ExclaimationQuote(Quote):
	"""docstring for ExclaimationQuoye"""
	def says(self):
		return self.words + '!'
class BabblingBrook():
 	def who(self):
 		return 'Brook'
 	def says(self):
 		return 'Babble'
hunter = Quote('Elmer Fudd', "I'm hunting wabbits")
hunted1 = QuestionQuote('Bugs Bunny', "What's up, doc")
hunted2 = ExclaimationQuote('Daffy Duck', "It's rabbit season")
brook = BabblingBrook() 
def who_says(obj):
	print(obj.who(), 'says:', obj.says())
who_says(brook)
who_says(hunter)
who_says(hunted1)
who_says(hunted2)
print('----------------------------')
class Word():
	"""docstring for Word"""
	def __init__(self, text):
		self.text = text
	def equals(self, word2):
		return self.text.lower() == word2.text.lower()
first = Word('ha')
second = Word('HA')
third = Word('eh')
print(first.equals(second))
print(first.equals(third))
print('----------------------------')
class Word():
	"""docstring for Word"""
	def __init__(self, text):
		self.text = text
	def __eq__(self, word2):
		return self.text.lower() == word2.text.lower()
first = Word('ha')
second = Word('HA')
third = Word('eh')
print(first == second)
print(first == third)
print('----------------------------')
class Word():
	"""docstring for Word"""
	def __init__(self, text):
		self.text = text
	def __eq__(self, word2):
		return self.text.lower() == word2.text.lower()
	def __str__(self):
		return self.text
	def __repr__(self):
		return 'Word('" self.text "')'
first = Word('ha')
print(first)
print('----------------------------')
class Bill():
	"""docstring for Bill"""
	def __init__(self, description):
		self.description = description
		
class Tail():
	"""docstring for Tail"""
	def __init__(self, length):
		self.length = length
		
class Duck():
	"""docstring for Duck"""
	def __init__(self, bill, tail):
		self.bill = bill
		self.tail = tail
	def  about(self):
		print('This duck has a', bill.description, 'bill and a', tail.length, 'tail')
tail = Tail('long')
bill = Bill('wide orange')
duck = Duck(bill, tail)
duck.about()
print('----------------------------')
from collections import namedtuple
Duck = namedtuple('Duck', 'bill tail')
duck = Duck('wide orange', 'long')
print(duck)
print(duck.bill)
print(duck.tail)
print('----------------------------')
parts = {'bill': 'wide orange', 'tail': 'long'}
duck2 = Duck(**parts)
print(duck2)
print('----------------------------')
duck3 = duck2._replace(tail = 'magnificent', bill = 'crushing')
print(duck3)
print('----------------------------')
class Laser():
	def does(self):
		return 'disintegrate'
class Claw():
	def does(self):
		return 'crush'
class SmartPhone():
	def does(self):
		return 'ring'
class Robot():
	def __init__(self):
		self.laser = Laser()
		self.claw = Claw()
		self.smartphone = SmartPhone()
	def does(self):
		return '''I have many attachments:
		My laser, to %s.
		My claw, to %s.
		My smartphone, to %s.'''% (self.laser.does(), self.claw.does(), self.smartphone.does())
robbie = Robot()
print(robbie.does())
#2016-8-22
import unicodedata
def unicode_test(value):
	import unicodedata
	name = unicodedata.name(value)
	value2 = unicodedata.lookup(name)
	print('value = "%s", name = "%s", value2 = "%s"' %(value, name, value2))
unicode_test('A')
unicode_test('$')
unicode_test('\u00a2')
unicode_test('\u20ac')
unicode_test('\u2603')
print(unicodedata.name('\u00e9'))
print(unicodedata.lookup('LATIN SMALL LETTER E WITH ACUTE'))
place = 'caf\u00e9'
print(place)
place = 'caf\N{LATIN SMALL LETTER E WITH ACUTE}'
print(place)
u_umlaut = '\N{LATIN SMALL LETTER U WITH DIAERESIS}'
drink = 'Gew' + u_umlaut +'ratraminer'
print('Now I can finally have my', drink, 'in a', place)
print(len('\U0001f47b'))
snowman = '\u2603'
print(len(snowman))
ds = snowman.encode('utf-8')
print(len(ds))
print(snowman.encode('ascii', 'ignore'))
print(snowman.encode('ascii', 'replace'))
print(snowman.encode('ascii', 'backslashreplace'))
print(snowman.encode('ascii', 'xmlcharrefreplace'))
print(ds)
place = 'caf\u00e9'
print("place:",place)
print(type(place))
place_bytes = place.encode('utf-8')
print(place_bytes)
print(type(place_bytes))
place2 = place_bytes.decode('utf-8')
print("place2:",place2)
place3 = place_bytes.decode('latin-1')
print('place3:', place3)
print('%s' % 42)
print('%d' %42)
print('%x' % 42)
print('%o' % 42)
print('%s' % 7.03)
print('%f' % 7.03)
print('%e' % 7.03)
print('%g' % 7.03)
print('%d%%' % 100)
actor = 'Richard Cere'
cat = 'Chester'
weight = 28
print("My wife's favorite actor is %s" % actor)
print("Our cat %s weight %s pounds" % (cat, weight))
n = 42
f = 7.03
s = 'string cheese'
print('%d %f %s' % (n, f, s))
print('%10d %10f %10s' % (n, f, s))
print('%-10d %-10f %-10s' % (n, f, s))
print('%10.4d %10.4f %10.4s' % (n, f, s))
print('%.4d %.4f %.4s' % (n, f, s))
print('%*.*d %*.*f %*.*s' % (10, 4, n, 10, 4, f, 10, 4, s))
print('----------------------------')
print('{} {} {}'.format(n, f, s))
print('{2} {0} {1}'.format(n, f, s))
print('{n} {f} {s}'.format(n=42, f=7.03, s='string cheese'))
d = {'n': 42, 'f': 7.03, 's': 'string cheese'}
print('{0[n]} {0[f]} {0[s]} {1}'.format(d, 'other'))
print('{0:d} {1:f} {2:s}'.format(n, f, s))
print('{n:d} {f:f} {s:s}'.format(n=42, f=7.03, s='string cheese'))
print('{0:10d} {1:10f} {2:10s}'.format(n, f, s))
print('{0:>10d} {1:>10f} {2:>10s}'.format(n, f, s))
print('{0:<10d} {1:<10f} {2:<10s}'.format(n, f, s))
print('{0:>10d} {1:>10.4f} {2:>10.4s}'.format(n, f, s))
print('{0:!^20s}'.format('BIG SALE'))
import re
source = 'Young Frankenstein'
m = re.match('You', source)
if m:
	print(m.group())
else:
	print('NOT MATCH')
m =re.search('Frank', source)
print('----------------------------')
if m:
	print(m.group())
else:
	print('NOT MATCH')
m = re.match('.*Frank', source)
print('----------------------------')
if m:
	print(m.group())
else:
	print('NOT MATCH')
m = re.search('Frank', source)
print('----------------------------')
if m:
	print(m.group())
else:
	print('NOT MATCH')
m = re.findall('n', source)
print('----------------------------')
print(m)
print('Found', len(m), 'matches')
m = re.findall('n.', source)
print('----------------------------')
print(m)
print('Found', len(m), 'matches')
m = re.findall('n.?', source)
print('----------------------------')
print(m)
print('Found', len(m), 'matches')
m = re.split('n', source)
print('----------------------------')
print(m)
print('Found', len(m), 'matches')
m = re.sub('n', '?', source)
print('----------------------------')
print(m)
print('----------------------------')
import string, re
printable = string.printable
print(len(printable))
print(printable[0:50])
print(printable[50:])
print(re.findall('\d', printable))
print(re.findall('\w', printable))
print(re.findall('\s', printable))
x = 'abc' + '-/*' + '\u00ea' + '\u0115'
print(re.findall('\w', x))
print('----------------------------')
source = 'I wish I may, I wish I might have a dish of fish tonight.'
print(re.findall('wish', source))
print(re.findall('wish|fish', source))
print(re.findall('^wish', source))
print(re.findall('^I wish', source))
print(re.findall('fish$', source))
print(re.findall('fish tonight.$', source))
print(re.findall('fish tonight\.$', source))
print(re.findall('[wf]ish', source))
print(re.findall('[wsh]+', source))
print(re.findall('ght\W', source))
print(re.findall('I (?=wish)', source))
print(re.findall('(?<=I) wish', source))
print(re.findall(r'\bfish', source))
import re
source = 'I wish I may, I wish I might have a dish of fish tonight.'
m = re.search(r'(. dish\b).*(\bfish)', source)
print(m.group())
print(m.groups())
m = re.search(r'(?P<DISH>. dish\b).*(?P<FISH>\bfish)', source)
print(m.group())
print(m.groups())
print(m.group('DISH'))
print(m.group('FISH'))
#2016-8-24
blist = [1, 2, 3, 255]
print(bytes(blist))
print(bytearray(blist))
print(b'\x61')
print(b'\x01abc\xff')
the_byte_array = bytearray(blist)
print(the_byte_array)
the_byte_array[1] = 127
print(the_byte_array)
print(bytes(range(0, 256)))
print(bytearray(range(0, 256)))
import struct
valid_png_header = b'\x89PNG\r\n\x1a\n'
data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR' + b'\x00\x00\x00\x9a\x00\x00\x00\x8d\x08\x02\x00\x00\x00\xc0'
if data[:8] == valid_png_header:
    width, height = struct.unpack('>LL', data[16:24])
    print('Valid PNG, width:', width, ',', 'height:', height)
else:
    print('Not a valid PNG')
print(data[16:20])
print(data[20:24])
import struct
print(struct.pack('>L', 254))
print(struct.pack('<L', 254))
import binascii
valid_png_header = b'\x89PNG\r\n\x1a\n'
print(binascii.hexlify(valid_png_header))
print(binascii.unhexlify(b'89504e470d0a1a0a'))
a = 60            # 60 = 0011 1100
b = 13            # 13 = 0000 1101
c = 0
c = a & b;        # 12 = 0000 1100
print("Line 1 - Value of c is ", c)
c = a | b;        # 61 = 0011 1101
print("Line 2 - Value of c is ", c)
c = a ^ b;        # 49 = 0011 0001
print("Line 3 - Value of c is ", c)
c = ~a;           # -61 = 1100 0011
print("Line 4 - Value of c is ", c)
c = a << 2;       # 240 = 1111 0000
print("Line 5 - Value of c is ", c)
c = a >> 2;       # 15 = 0000 1111
print("Line 6 - Value of c is ", c)
import unicodedata
mystery = '\U0001f4a9'
print(mystery)
print(unicodedata.name(mystery))
pop_bytes = bytes(mystery.encode('utf-8'))
print(pop_bytes)
pop_string = pop_bytes.decode('utf-8')
print(pop_string)
print("My kitty cat likes %s,\nMy kitty cat likes %s,\nMy kitty cat fell on his %s,\nAnd now thinks he's a %s." % ( 'roast beef', 'ham', 'head', 'clam'))
letter = '''
    Dear {salutation} {name},

    Thank you for your letter. We are sorry that our {product} {verbed} in your {room}. Please note that it should never be used in a {room}, especially near any {animals}.

    Send us your receipt and {amount} for shipping and handling. We will send you another {product} that, in our tests, is {percent}% less likely to have {verbed}.

    Thank you for your support.

    Sincerely,
    {spokesman}
    {job_title}
'''
response = {'salutation': 'HL', 'name': 'Eve', 'product': 'car', 'verbed': 'run', 'room': 'yard', 'animals': 'rabbits', 'amount': '20', 'percent': '20', 'spokesman': 'Barak', 'job_title': 'Artist'}
print(letter.format(**response))
#import!!!
import re
mammoth = '''
    We have seen thee, queen of cheese, 
    Lying quietly at your ease, 
    Gently fanned by evening breeze, 
    Thy fair form no flies dare seize.

    All gaily dressed soon you'll go
    To the great Provincial show,
    To be admired by many a beau
    In the city of Toronto.

    Or as the leaves upon the trees, 
    It did require to make thee please, 
    And stand unrivalled, queen of cheese.

    May you not receive a scar as 
    We have heard that Mr. Harris 
    Intends to send you off as far as 
    The great world's show at Paris.

    Of the youth beware of these, 
    For some of them might rudely squeeze 
    And bite your cheek, then songs or glees 
    We could not sing, oh! queen of cheese.

    We'rt thou suspended from balloon, 
    You'd cast a shade even at noon, 
    Folks would think it was the moon 
    About to fall and crush them soon.
'''
print(re.findall(r'\bc\w*', mammoth))
print(re.findall(r'\bc\w{3}\b', mammoth))
print(re.findall(r'\b\w*r\b', mammoth))
print(re.findall(r'\b\w*[aeiou]{3}[^aeiou\s]*\w*\b', mammoth))
#2016-8-25
import binascii, struct
gif = binascii.unhexlify(b'47494638396101000100800000000000ffffff21f90401000000002c000000000100010000020144003b' )
print(gif)
if gif[:6] == b'GIF89a':
    print('VALID GIF!')
    width, height = struct.unpack('<HH', gif[6:10])
    print("width:", width, 'height:', height)
else:
    print('NOT A VALID GIF')
poem = '''
There was a young lady named Bright,
Whose speed was far faster than light;
She started one day
In a relative way,
And returned on the previous night.
'''
print(len(poem))
fout = open('relativity.txt', 'wt')
fout.write(poem)
fout.close()
fout = open('relativity.txt', 'wt')
print("There was a young lady named Bright,", "Whose speed was far faster than light;", "She started one day", "In a relative way,", "And returned on the previous night.", file = fout, sep = '$', end = '')
fout.close()
fout.close()
try:
    fout = open('relativity.txt', 'xt')
    fout.write('stomp stomp stomp')
except FileExistsError:
    print('relativity already exists! That was a close one.')
fin = open('relativity.txt', 'rt')
poem = fin.read()
fin.close()
print(len(poem))
print(poem)
poem = ''
fin = open('relativity.txt', 'rt')
chunk = 100
while True:
    fragment = fin.read(chunk)
    if not fragment:
        break
    poem += fragment
fin.close()
print(len(poem))
print(poem)
poem = ''
fin = open('relativity.txt', 'rt')
while True:
    line = fin.readline()
    if not line:
        break
    poem += line
fin.close()
print(len(poem))
print(poem)
poem = ''
fin = open('relativity.txt', 'rt')
for line in fin:
    poem += line
fin.close()
print(len(poem))
print(poem)
fin = open('relativity.txt', 'rt')
lines = fin.readlines()
fin.close()
print(len(lines), 'lines read')
for line in lines:
    print(line, end = '')
bdata = bytes(range(0, 256))
print(len(bdata))
fout = open('bfile', 'wb')
fout.write(bdata)
fout.close()
fout = open('bfile', 'wb')
size = len(bdata)
offset = 0
chunk = 100
while True:
    if offset > size:
        break
    fout.write(bdata[offset: offset + chunk])
    offset += chunk
fin = open('bfile', 'rb')
bdata = fin.read()
print(len(bdata))
fin.close()
with open('relativity.txt', 'wt') as fout:
	fout.write(poem)
fin = open('bfile', 'rb')
print(fin.tell())
print(fin.seek(255))
bdata = fin.read()
print(len(bdata))
print(bdata[0])
import os
print(os.SEEK_SET)
print(os.SEEK_CUR)
print(os.SEEK_END)
fin = open('bfile', 'rb')
print(fin.seek(-1, 2))
print(fin.tell())
bdata = fin.read()
print(len(bdata))
print(bdata[0])
print(fin.seek(254, 0))
print(fin.tell())
print(fin.seek(1, 1))
print(fin.tell())
bdata = fin.read()
print(len(bdata))
print(bdata[0])
fin.close()
import csv
villains = [
    ['Doctor', 'No'],
    ['Rosa', 'Klebb'],
    ['Mister', 'Big'],
    ['Auric', 'Goldfinger'],
    ['Ernst', 'Blofeld'],
]
with open('villains', 'wt') as fout:
    csvout = csv.writer(fout)
    csvout.writerows(villains)
import csv
with open('villains', 'rt') as fin:
    cin = csv.reader(fin)
    villains = [row for row in cin]
print(villains)
import csv
with open('villains', 'rt') as fin:
    cin = csv.DictReader(fin, fieldnames=['first', 'last'])
    villains = [row for row in cin]
print(villains)
import csv
villains = [
    {'first': 'Doctor', 'last': 'No'},
    {'first': 'Rosa', 'last': 'Klebb'},
    {'first': 'Mister', 'last': 'Big'},
    {'first': 'Auric', 'last': 'Goldfinger'},
    {'first': 'Ernst', 'last': 'Blofeld'}
]
with open('villains', 'wt') as fout:
    cout = csv.DictWriter(fout, ['first', 'last'])
    cout.writeheader()
    cout.writerows(villains)
import csv
with open('villains', 'rt') as fin:
    cin = csv.DictReader(fin)
    villains = [row for row in cin]
print(villains)
input('press anything to continute...')
