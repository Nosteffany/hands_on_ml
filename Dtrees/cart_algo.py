from typing import ClassVar


class TestClass:
    CLASSVAR_1 = 10
    CLASSVAR_2 = 100
    instances = 0

    
    def __init__(self) -> None:
        self.state = dict()
        __class__.instances += 1
    
    
    def update_state(self, key: str, value: float ):
        self.state.update(dict(key,value))

    
    
    def get_classvar(self):
        print(__class__.CLASSVAR_1)

obj1 = TestClass()
obj2 = TestClass()

print(obj1.instances, obj2.instances)
print(TestClass.instances)

