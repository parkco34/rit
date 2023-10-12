class Toy(object):  # Toy is a subclass of object
    def __init__(self):
        self._elems = []
        
    def add(self, new_elems):
        """New elements list"""
        self._elems += new_elems
        
    def size(self):
            return len(self._elems)
        
    def __str__(self):
        """ Without this, the output would be: 
        Value of t3 is  <__main__.Toy object at 0x7f11f16fad90> """
        return str(self._elems)
        
    def __add__(self, other):
        new_toy = Toy()
        new_toy._elems = self._elems + other._elems
        return new_toy
    
    def __len__(self):
        """ Returns the number of elements in self """
        return len(self._elems)
    
    def __eq__(self, other):
        return self._elems == other._elems
    
    def __hash__(self):
        return id(self)
    
    
class Int_set(object):
    """ An Int_set is a set of integers """
    # Information about the implementation (not the abstraction)
    #  - The value of a set is represented by a list of ints, self._vals.
    #  - Each int in self._vals occurs exactly once.
    
    def __init__(self):
        """ Creates an empty set of integers """
        self._vals = []
        
    def insert(self, e):
        """ Assumes e is an integer and inserts e into self """
        if not e in self._vals:
            self._vals.append(e)
            
    def member(self, e):
        """ Assumes e is an integer
            Returns True if e is in self, and False otherwise """
        return e in self._vals
    
    def remove(self, e):
        """ Assumes e is an integer and removes e from self
            Raises ValueError if e is not in self """
        try:
            self._vals.remove(e)
        except:
            raise ValueError(str(e) + ' not found')
        
    def get_members(self):
        """ Returns a list containing the elements of self._vals.
            Nothing can be assumed about the order of the elements """
        return self._vals[:]
    
    def union(self, other):
        """ Assumes other is an Int_set
            Returns a new Int_set representing the union of the elements in self and other """
        # Ensure other is an Int_set
        if isinstance(other, Int_set):
            for el in other.get_members():
                print(el)
                if el not in self._vals:
                    self.insert(el)
                    
    # Finger exercise 2
    def __add__(self, other):
        """ Method that allows client of Int_set to use the + operator to denote set union"""
        if isinstance(self, other):
            union_set = Int_set()
            
            for el in self._vals:
                union_set.insert(el)
                
            for el in other.get_members():
                union_set.insert(el)
                
            return union_set
        
        else:
            raise TypeError("Other must be in instance of Int_set")
    
    def __str__(self):
        """ Returns a string representation of self """
        """ (If no __str__ method were defined, executing print(s) would
        cause something like <__main__.Int_set object at 0x1663510> to be
        printed.)"""
        self._vals.sort()
        result = ''
        for e in self._vals:
            result = result + str(e) + ','
        return '{' + result[:-1] + '}'
        
    
    
def main():
    print(type(Toy))
    print(type(Toy.__init__), type(Toy.add), type(Toy.size))
    # Toy.size is an ATTRIBUTE of the class Toy
    t1 = Toy()   # When executed, interpreter creates new INSTANCE of type Toy
    print(type(t1))
    print(type(t1.add))
    t2 = Toy()
    print(t1 is t2)
    t1.add([1, 2])
    t2.add([3, 4])
    print(t2)
    t3 = t1 + t2
    print("Value of t3 is ", t3)
    print("The length of t3 is ", len(t3))
    d = {t1: 'A', t2: 'B'}
    print("The value ", d[t1], " is associated with t1 in dictionary d")
    # Had we not created __hash__ method, the following would have raised an error

    
    
    s = Int_set()
    s.insert(3)
    print(s.member(3))
    # Finger exercise 1
    other = Int_set()
    other.insert(5), other.insert(7), other.insert(9), other.insert(11)
    s.union(other)
    print(s)


if __name__ == "__main__":
    main()