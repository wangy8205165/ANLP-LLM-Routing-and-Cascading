def divide(a, b):
    assert b != 0, "除数不能为 0"
    return a / b

print(divide(10, 2))  # ✅
print(divide(10, 0))  # ❌ 抛出 AssertionError