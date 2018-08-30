# def constrained_compositions(n, m):
#     # inputs: n is of type 'int' and m is a list of integers
#     # output: a set of tuples
#
#     k = len(m)
#     parts = set()
#     if k == n:
#         if 1 <= min(m):
#             parts.add((1,) * n)
#     if k == 1:
#         if n <= m[0]:
#             parts.add((n,))
#     else:
#         for x in range(1, min(n - k + 2, m[0] + 1)):
#             for y in constrained_compositions(n - x, m[1:]):
#                 parts.add((x,) + y)
#     return parts
#
#
# print(constrained_compositions(7, [1, 2, 3, 4]))

try:
    msg = 'reinforcement learning'
    times = int(input(f'how many time "{msg}" to display? '))
    if times < 0:
        raise Exception('number should be positive')
    for i in range(times):
        print(msg)
except Exception as e:
    print(e)



