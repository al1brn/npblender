class Root:
    __slots__ = ('names', 'x', 'y')
    _slots = ('names', 'x', 'y')

    def __init__(self, x, y):
        self.names = {'attr': "YES"}
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"
    
    def __setattr__(self, name, value):
        if name in self._slots:
            super().__setattr__(name, value)
        else:
            self.names[name] = value
    
    def __getattr__(self, name):
        if name in self.names:
            return self.names[name]
        raise AttributeError(f"Unknown attribute: '{name}'")

class Child(Root):
    __slots__ = ('z',)
    _slots = Root._slots + ('z',)

    def __init__(self, x, y, z):
        super().__init__(x, y)  
        self.z = z

    def __str__(self):
        return super().__str__() + f", {self.z}"
    


a = Root(1, 2)
b = Child(1, 2, 3)

a.test1 = "Test1"
b.test2 = "Test2"

print("Root: ", a, a.test1)
print("Child:", b, b.test2)




