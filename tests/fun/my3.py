
def sum(n: int, acc: int) -> int:
    if n == 0:
        return acc
    else:
        return sum(n-1, acc+n)

print(sum(1, 0))