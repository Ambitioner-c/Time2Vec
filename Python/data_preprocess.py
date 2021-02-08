import csv


def read_csv(pathname):
    data_dict = dict()
    with open(pathname, 'r') as f:
        num = 0
        for j in csv.reader(f):
            if num is not 0:
                try:
                    data_dict[j[1]] = list(map(float, j[2:]))
                except ValueError as e:
                    continue
            num += 1

    return data_dict


def write_csv(_dict, pathname='../Data/all.csv'):
    with open(pathname, 'w') as f:
        writer = csv.writer(f)
        for j in _dict:
            _list = [j]
            _list.extend(_dict[j])
            writer.writerow(_list)


def preprocess(data_dict):
    new_data_dict = dict()
    for j in data_dict:
        # delete zero lines
        data_list = data_dict[j]
        _sum = sum(data_list)
        length = len(data_list)
        avg = _sum / length
        if avg == 0.0:
            continue

        # replace big exceptional value
        exceptional_list_big = []
        number_big = 0
        # replace negative value
        exceptional_list_neg = []
        number_neg = 0
        # number of zero
        number_zero = 0
        for k in data_list:
            # replace big exceptional value
            if k > 10000.0:
                exceptional_list_big.append(number_big)
                _sum = _sum - k
            elif k < -10000.0:
                exceptional_list_big.append(number_big)
                _sum = _sum + k
            number_big += 1
            # replace negative value
            if k < 0.0:
                exceptional_list_neg.append(number_neg)
            number_neg += 1
            # number of zero
            if k == 0.0:
                number_zero += 1
        avg = _sum / length
        # replace big exceptional value
        for k in exceptional_list_big:
            data_list[k] = data_list[k-1]
        # replace negative value
        for k in exceptional_list_neg:
            data_list[k] = data_list[k-1]
        # number of zero
        if number_zero > 30:
            continue

        # replace mean
        number_mean = 0
        for k in data_list:
            if k >= 2 * avg:
                if data_list[number_mean - 1] == 0.0:
                    if data_list[number_mean - 2] == 0.0:
                        if data_list[number_mean - 3] == 0.0:
                            if data_list[number_mean - 4] == 0.0:
                                if data_list[number_mean - 5] == 0.0:
                                    if data_list[number_mean - 6] == 0.0:
                                        if data_list[number_mean - 7] == 0.0:
                                            pass
                                        else:
                                            data_list[number_mean - 6] = round(k / 7, 1)
                                            data_list[number_mean - 5] = round(k / 7, 1)
                                            data_list[number_mean - 4] = round(k / 7, 1)
                                            data_list[number_mean - 3] = round(k / 7, 1)
                                            data_list[number_mean - 2] = round(k / 7, 1)
                                            data_list[number_mean - 1] = round(k / 7, 1)
                                            data_list[number_mean - 0] = round(k / 7, 1)
                                    else:
                                        data_list[number_mean - 5] = round(k / 6, 1)
                                        data_list[number_mean - 4] = round(k / 6, 1)
                                        data_list[number_mean - 3] = round(k / 6, 1)
                                        data_list[number_mean - 2] = round(k / 6, 1)
                                        data_list[number_mean - 1] = round(k / 6, 1)
                                        data_list[number_mean - 0] = round(k / 6, 1)
                                else:
                                    data_list[number_mean - 4] = round(k / 5, 1)
                                    data_list[number_mean - 3] = round(k / 5, 1)
                                    data_list[number_mean - 2] = round(k / 5, 1)
                                    data_list[number_mean - 1] = round(k / 5, 1)
                                    data_list[number_mean - 0] = round(k / 5, 1)
                            else:
                                data_list[number_mean - 3] = round(k / 4, 1)
                                data_list[number_mean - 2] = round(k / 4, 1)
                                data_list[number_mean - 1] = round(k / 4, 1)
                                data_list[number_mean - 0] = round(k / 4, 1)
                        else:
                            data_list[number_mean - 2] = round(k / 3, 1)
                            data_list[number_mean - 1] = round(k / 3, 1)
                            data_list[number_mean - 0] = round(k / 3, 1)
                    else:
                        data_list[number_mean - 1] = round(k / 2, 1)
                        data_list[number_mean - 0] = round(k / 2, 1)

        new_data_dict[j] = data_list

    return new_data_dict


if __name__ == '__main__':
    _pathname = '../DataPreprocess/All.csv'
    _data_dict = read_csv(_pathname)
    print(len(_data_dict))

    __data_dict = preprocess(_data_dict)
    write_csv(__data_dict)
    print(len(__data_dict))

