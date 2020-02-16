import numpy as np

def print_field(field):
    print('+' + '-' * 8 + '+')
    for i in range(8):
        cur = '|'
        for j in range(8):
            if field[i][j] == 1:
                cur += 'D'
            elif field[i][j] == 2:
                cur += 'd'
            elif field[i][j] == 3:
                cur += 's'
            else:
                cur += '`'
        cur += '|'
        print(cur)
    print('+' + '-' * 8 + '+')


def sheep_can_move(field, sh):
    move = {'u': (-1, 0), 'l': (0, -1), 'd': (1, 0), 'r': (0, 1)}
    for key in ['u', 'l', 'd', 'r']:
        pend_pos = sh + move[key]
        if pend_pos[0] not in range(8) or pend_pos[1] not in range(8) or field[pend_pos[0]][pend_pos[1]] > 0:
            move.pop(key)
    can_move = list(move.keys())
    return move, can_move


def sheep_move(field, sh):
    move, can_move = sheep_can_move(field, sh)
    if len(can_move) == 0:
        return field, sh
    choice = move[can_move[np.random.randint(0, len(can_move))]]
    field[sh[0]][sh[1]] = 0
    sh += choice
    field[sh[0]][sh[1]] = 3
    return field, sh


def manhattan_distance(pos1, pos2):
    d = abs(pos1[0] - pos2[0])
    d += abs(pos1[1] - pos2[1])
    return d


def dogs_move(field, d1, d2, sh):
    sh_move, sh_can = sheep_can_move(field, sh)
    if sh[0] == 7:
        if sh[1] == 7:
            choice = [
                [[6, 6], [6, 7]],
                [[7, 6], [6, 6]]
            ]
        else:
            choice = [
                [[7, sh[1] + 1], [6, sh[1] + 1]]
            ]
    elif sh[1] == 7:
        choice = [
            [[sh[0] + 1, 6], [sh[0] + 1, 7]]
        ]
    else:
        choice = [
            [[sh[0] + 1, sh[1]], [sh[0], sh[1] + 1]]
        ]
    min_e = 20
    flag = 0
    for ch in choice:
        d1to0 = manhattan_distance(d1, ch[0])
        d1to1 = manhattan_distance(d1, ch[1])
        if d1[0] == ch[0][0] and d1[1] < ch[0][1] or d1[1] == ch[0][1] and d1[0] < ch[0][0]:
            d1to0 += 2
        if d1[0] == ch[1][0] and d1[1] < ch[1][1] or d1[1] == ch[1][1] and d1[0] < ch[1][0]:
            d1to1 += 2
        d2to0 = manhattan_distance(d2, ch[0])
        d2to1 = manhattan_distance(d2, ch[1])
        if d2[0] == ch[0][0] and d2[1] < ch[0][1] or d2[1] == ch[0][1] and d2[0] < ch[0][0]:
            d1to0 += 2
        if d2[0] == ch[1][0] and d2[1] < ch[1][1] or d2[1] == ch[1][1] and d2[0] < ch[1][0]:
            d1to1 += 2
        way1 = max([d1to0, d2to1])
        way2 = max([d1to1, d2to0])
        if way1 <= way2 and way1 < min_e:
            flag = 0
            d1target = ch[0]
            d2target = ch[1]
        elif way2 < way1 and way2 < min_e:
            flag = 1
            d1target = ch[1]
            d2target = ch[0]
    if flag == 0:
        if abs(d1[0] - d1target[0]) < abs(d1[1] - d1target[1]):
            if d1[1] < d1target[1]:
                if field[d1[0]][d1[1] + 1] == 0:
                    field[d1[0]][d1[1]] = 0
                    d1[1] += 1
                    field[d1[0]][d1[1]] = 1
            elif d1[1] > d1target[1]:
                if field[d1[0]][d1[1] - 1] == 0:
                    field[d1[0]][d1[1]] = 0
                    d1[1] -= 1
                    field[d1[0]][d1[1]] = 1
        else:
            if d1[0] < d1target[0]:
                if field[d1[0] + 1][d1[1]] == 0:
                    field[d1[0]][d1[1]] = 0
                    d1[0] += 1
                    field[d1[0]][d1[1]] = 1
            elif d1[0] > d1target[0]:
                if field[d1[0] - 1][d1[1]] == 0:
                    field[d1[0]][d1[1]] = 0
                    d1[0] -= 1
                    field[d1[0]][d1[1]] = 1

        if abs(d2[1] - d2target[1]) < abs(d2[0] - d2target[0]):
            if d2[0] < d2target[0]:
                if field[d2[0] + 1][d2[1]] == 0:
                    field[d2[0]][d2[1]] = 0
                    d2[0] += 1
                    field[d2[0]][d2[1]] = 2
            elif d2[0] > d2target[0]:
                if field[d2[0] - 1][d2[1]] == 0:
                    field[d2[0]][d2[1]] = 0
                    d2[0] -= 1
                    field[d2[0]][d2[1]] = 2
        else:
            if d2[1] < d2target[1]:
                if field[d2[0]][d2[1] + 1] == 0:
                    field[d2[0]][d2[1]] = 0
                    d2[1] += 1
                    field[d2[0]][d2[1]] = 2
            elif d2[1] > d2target[1]:
                if field[d2[0]][d2[1] - 1] == 0:
                    field[d2[0]][d2[1]] = 0
                    d2[1] -= 1
                    field[d2[0]][d2[1]] = 2
    else:
        if abs(d1[1] - d1target[1]) < abs(d1[0] - d1target[0]):
            if d1[0] < d1target[0]:
                if field[d1[0] + 1][d1[1]] == 0:
                    field[d1[0]][d1[1]] = 0
                    d1[0] += 1
                    field[d1[0]][d1[1]] = 1
            elif d1[0] > d1target[0]:
                if field[d1[0] - 1][d1[1]] == 0:
                    field[d1[0]][d1[1]] = 0
                    d1[0] -= 1
                    field[d1[0]][d1[1]] = 1
        else:
            if d1[1] < d1target[1]:
                if field[d1[0]][d1[1] + 1] == 0:
                    field[d1[0]][d1[1]] = 0
                    d1[1] += 1
                    field[d1[0]][d1[1]] = 1
            elif d1[1] > d1target[1]:
                if field[d1[0]][d1[1] - 1] == 0:
                    field[d1[0]][d1[1]] = 0
                    d1[1] -= 1
                    field[d1[0]][d1[1]] = 1

        if abs(d2[0] - d2target[0]) < abs(d2[1] - d2target[1]):
            if d2[1] < d2target[1]:
                if field[d2[0]][d2[1] + 1] == 0:
                    field[d2[0]][d2[1]] = 0
                    d2[1] += 1
                    field[d2[0]][d2[1]] = 2
            elif d2[1] > d2target[1]:
                if field[d2[0]][d2[1] - 1] == 0:
                    field[d2[0]][d2[1]] = 0
                    d2[1] -= 1
                    field[d2[0]][d2[1]] = 2
        else:
            if d2[0] < d2target[0]:
                if field[d2[0] + 1][d2[1]] == 0:
                    field[d2[0]][d2[1]] = 0
                    d2[0] += 1
                    field[d2[0]][d2[1]] = 2
            elif d2[0] > d2target[0]:
                if field[d2[0] - 1][d2[1]] == 0:
                    field[d2[0]][d2[1]] = 0
                    d2[0] -= 1
                    field[d2[0]][d2[1]] = 2

    return field, d1, d2


def pend_catch(d1, d2, sh):
    if (d1 == sh + [0, 1]).all() and (d2 == sh + [1, 0]).all() or (d1 == sh + [1, 0]).all() and (
            d2 == sh + [0, 1]).all():
        return True
    return False


def main():
    field = np.zeros(shape=(8, 8))
    dog1_pos = np.random.randint(0, 8, 2)
    field[dog1_pos[0]][dog1_pos[1]] = 1
    dog2_pos = np.random.randint(0, 8, 2)
    while (dog2_pos == dog1_pos).all():
        dog2_pos = np.random.randint(0, 8, 2)
    field[dog2_pos[0]][dog2_pos[1]] = 2
    sheep_pos = np.random.randint(0, 8, 2)
    while (sheep_pos == dog1_pos).all() and (sheep_pos == dog2_pos).all():
        sheep_pos = np.random.randint(0, 8, 2)
    field[sheep_pos[0]][sheep_pos[1]] = 3

    count = 0
    while not pend_catch(dog1_pos, dog2_pos, sheep_pos):
        field, dog1_pos, dog2_pos = dogs_move(field, dog1_pos, dog2_pos, sheep_pos)
        field, sheep_pos = sheep_move(field, sheep_pos)
        count += 1
    count += manhattan_distance(sheep_pos, [0, 0])
    return count


if __name__ == '__main__':
    total = 0
    for i in range(10000):
        total += main()
    
    print(total / 10000)
