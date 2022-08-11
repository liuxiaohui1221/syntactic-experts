# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import random


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    random.seed(100)
    n=random.randint(0, 100)
    print(n)
    n = random.randint(0, 100)
    print(n)

    a1=[1,2,3,4,5]
    b=a1[:2]+a1[3:]
    print(b)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
