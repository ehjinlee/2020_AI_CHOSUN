print("안녕하세요.")
test_a = 3
if test_a == 3:
    print("3")
else:
    print("3이 아니다")

for i in range(0,10):
    print(i)

temp = 0
while True:
    temp = temp + 1
    print(temp)
    if temp > 40:
        break

while temp < 60:
    temp += 1
    print(temp)

def add_function(x):
    y = x+3
    return y
print(add_function(3))


