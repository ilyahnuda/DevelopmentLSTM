import json
import matplotlib.pyplot as plt
import numpy as np


def draw_plot(title, X, y_pred):
    dates = [*zip(*X)][0]
    plt.plot(*zip(*X), label='Фактические')
    plt.plot(dates[-len(y_pred):], y_pred, 'g', label='Предсказанные')
    plt.title(title)
    plt.xlabel('Дата')
    plt.ylabel('Цена при закрытии торгов')
    plt.legend()
    plt.grid()
    plt.show()


def get_json_data(path: str):
    with open(path, 'r') as file:
        data = json.load(file)
    return data


def draw_line_charts_neurons(AVG_ERROR, NUMBER_NEURONS, TYPE_ERROR):
    if TYPE_ERROR == 'MAPE':
        type_line = 'm--'
    else:
        type_line = 'g--'
    plt.plot(NUMBER_NEURONS, AVG_ERROR, type_line, marker='o', markersize=6)
    plt.xlabel(f'Количество нейронов', fontsize=18)
    plt.ylabel(f'Средняя ошибка {TYPE_ERROR}', fontsize=16)
    plt.title(F'Зависимость {TYPE_ERROR} от количества нейронов')
    plt.grid()
    plt.show()


def draw_box_plot(neurons, error_data, error, title, xtitle):
    if error == 'MAPE':
        color = 'purple'
    else:
        color = 'green'
    box = plt.boxplot(error_data, patch_artist=True)
    plt.xticks(range(1, len(neurons) + 1), neurons)
    plt.xlabel(f'{xtitle}', fontsize=18)
    plt.ylabel(f'{error}', fontsize=16)
    plt.title(f'{title}')
    for patch in box['boxes']:
        patch.set_facecolor(color)
    plt.show()


def main():
    errors = ['MAPE', 'RMSE']
    data = get_json_data('my7.json')
    values_error = {'MAPE': {'values': [], 'layers': []}, 'RMSE': {'values': [], 'layers': []}}
    for i in range(1, 3):
        ONE_LAYER = [x for x in data if len(x['architecture']) == i]
        if i == 1:
            neurons = [x['architecture'][0] for x in ONE_LAYER]
        else:
            neurons = [f"({x['architecture'][0]},{x['architecture'][1]})" for x in ONE_LAYER]

        for error in errors:
            values = [[fl_ch(j) for j in x[error]['VALUES']] for x in ONE_LAYER]
            values_error[error]['values'].append([s for x in values for s in x])
            values_error[error]['layers'].append(i)
            avg_errors = [np.average(p) for p in values]

            draw_line_charts_neurons(avg_errors, neurons, error)

            draw_box_plot(neurons, values, error, f'Изменение {error} в зависимости от количества нейронов',
                          'Количество нейронов')
            print('MIN:', np.min(values))
            print('MAX:', np.max(values))
            print('STD:', np.std(values))
    for i in values_error.keys():
        draw_box_plot(values_error[i]['layers'], values_error[i]['values'], i,
                      f'Зависимость {i} от количества слоев LSTM', 'Количество слоев LSTM')


def fl_ch(i):
    return float('{:.4f}'.format(float(i)))


if __name__ == '__main__':
    main()
