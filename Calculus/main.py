import addition
import subtraction


def very_interesting_calculus_func(a,b, choice):
    if choice == 1:
        sum = addition.sum(a,b)
        print("Ваша сумма: ", sum)
    elif choice == 2:
        sub = subtraction.sub(a,b)
        print("Ваша разность: ", sub)

choice = int(input("Что будем делать, командир? 1 - сложение,2 - вычитание: "))
a = int(input("Введите первое число: "))
b = int(input("Введите первое число: "))
very_interesting_calculus_func(a,b,choice)