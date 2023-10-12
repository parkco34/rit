from  datetime import date

class Person(object):
    def __init__(self, name):
        """ Assumes name a string. Creates a person"""
        self._name = name
        self._birthday = None
        
        try:
            last_blank = name.rindex(' ')
            self._last_name = name[last_blank + 1:]
            
        except:
            self._last_blank = name
            
        self.birthday = None
        
    def get_name(self):
        """ Returns self's full name"""
        return self._name
    
    def get_last_name(self):
        """ Returns self's last name"""
        return self._last_name
    
    def set_birthday(self, birthdate):
        """ Assumes birthdate is of type date
        Sets self's birthday to birthdate"""
        self._birthday = birthdate
        
    def get_age(self):
        """ Returns self's current age in days"""
        if self._birthday == None:
            raise ValueError
        return (date.today() - self._birthday).days

    # Overloads the < operator 
    # This overloading provides automatic acces to any polymorphic method defined using __lt__
    def __lt__(self, other):
        """ ASsume other Person
        Returns True if self precedes other in alphabetical
        order, False otherwise. Comparison is based on last names
        , but if these are the same full names are compared."""
        if self._last_name == other._last_name:
            return self._name < other._name
        
        return self._last_name < other._last_name

    def __str__(self):
        """ Returns self's name"""
        return self._name
    
    
class UTSA_person(Person):
    _next_id_num = 0 # Identification number; class variable (Class Attribute)
    
    def __init__(self, name):
        super().__init__(name) # Call the superclass's _init__ method
        self._id_num = UTSA_person._next_id_num # Initialized using class attribute
        UTSA_person._next_id_num += 1
        
    def get_id_num(self):
        return self._id_num # Instance of UTSA_person
    
    def is_student(self):
        return isinstance(self, Student)
    
    # def is_student(self):
    """ Not necessary"""
    #     return type(self) == Grad or type(self) == UG
    
    def __lt__(self, other):
        return self._id_num < other._id_num
    
    
class Politician(Person):
    def __init__(self, name, party=None):
        super().__init__(name)
        """ Name and party are strings"""
        try:
            if not name:
                self.name = input("Enter your stupid name: ")
                
            else:
                self.name = name
                
            if not party:
                self.party = "the self!"
                
            else:
                self.party = party
            
        except Exception as e:
            print(" OOps! ", e)
        
    def get_party(self):
        """ Returns politcal party"""
        return self.party
        
    def might_agree(self, other):
        """ Returns true if self and other belong to same party"""
        other_party = other.get_party()
        if self.party == other_party:
            return True
        
        else:
            return False
        
    def __str__(self):
        return str(self.name)
    
    
class Student(UTSA_person):
    pass


class UG(Student):
    def __init__(self, name, class_year):
        super().__init__(name)
        self._year = class_year
        
    def get_class(self):
        return self._year
    
    
class Grad(Student):
    pass


class Transfer_student(Student):
    def __init__(self, name, from_school):
        UTSA_person.__init__(self, name)
        self._from_school = from_school
        
    def get_old_school(self):
        return self._from_school


# def main():
#     me = Person("Parkdaddy")
#     him = Person("David Parker")
#     her = Person("Jessica Parker")
#     print(him._name[:him._name.rindex(' ')] + ' ' + him.get_last_name())
#     me.set_birthday(date(1985, 6, 8))
#     him.set_birthday(date(1988, 4, 26))
#     her.set_birthday(date(1982, 9, 24))
#     print(me.get_name(), 'is ', me.get_age(), ' days old.')
#     print(him.get_name(), 'is ', him.get_age(), ' days old.')
#     print(her.get_name(), 'is ', her.get_age(), ' days old.')
    
#     # USTA Persons
#     p1 = UTSA_person('Parkdaddy')
#     p2 = UTSA_person('David Parker')
#     p3 = UTSA_person('Jessica Parker')
#     p4 = UTSA_person('Henri Parker')
#     print("p1 < p2 = ", p1 < p2)
#     print("p3 < p2 = ", p3 < p2)
#     print("p4 < p1 = ", p4 < p1)
#     print("p1 < p4 = ", p1 < p4)
#     # p1 < p2 is shorthand for p1.__lt__(p2)
#     print(str(p1) + '\'s id number is ' + str(p1.get_id_num()))
    
#     # Politicians
#     lincoln = Politician("Abraham Lincoln", "Republican")
#     trump = Politician("Donald Trump")
#     print(lincoln , " was obviously a" , lincoln.get_party())
#     print(trump, " is obviously not patriotic and is all about ", trump.get_party())
#     print("lincoln and trump agree with eachother:", lincoln.might_agree(trump))
    
#     # More Students
#     p5 = Grad("Henri Poincare")
#     p6 = UG("Some bum", 2003)
#     print(p5, "is a graduate student is ", type(p5) == Grad)
#     print(p5, "is an undergraduate student is ", type(p5) == UG)
#     print(p5, "is a student is", p5.is_student()) # is_student is a UTSA_person method but still works on subclasses
#     print(p6, "is a student is", p6.is_student())
#     # p6 is bound to object of type UG, not Student, but since UG is a subclass of 
#     # Student, the object to which p6 is bound is an instance of class Student (as well as an instance of UTSA_person and Person)
    
#     # Finger exercise:
#     print("isinstance("abs", str) == isinstance(str, str) ", isinstance("abs", str) == isinstance(str, str))
    
# if __name__ == "__main__":
#     main()