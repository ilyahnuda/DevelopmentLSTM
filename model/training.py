import json

import numpy as np

from DB.DBConfig import DBConfig
from DB.repositories.CompanyRep import CompanyRep
from DB.repositories.StockIndexRep import StockRep
from DB.Db import Database
from data_preprocess import DataSet
from model import StockPrediction, load_model
from ModelGenerator import ModelGenerator
from data_analysis import draw_plot
import matplotlib.pyplot as plt


def load_data(symbol_id):
    stockRep = StockRep(db)
    data = stockRep.get_stock_index(symbol_id)
    return data


def get_data_from_companies(count_companies):
    companyRep = CompanyRep(db)
    companies = companyRep.get_all_company_symbol()

    ALL_data = np.empty(shape=(0, 6))
    ALL_X_train = np.empty(shape=(0, window_size, 5))
    ALL_Y_train = np.empty(shape=(0, 5))
    ALL_X_test = np.empty(shape=(0, window_size, 5))
    ALL_Y_test = np.empty(shape=(0, 5))

    for company in companies[:count_companies]:
        data = load_data(company.id)
        data = data[-1000:]
        ALL_data = np.concatenate([ALL_data, data])

    dataset = DataSet(ALL_data)
    X_train, Y_train, X_test, Y_test = dataset.preprocess(ALL_data, 0.2)

    ALL_X_train = np.concatenate([ALL_X_train, X_train])
    ALL_Y_train = np.concatenate([ALL_Y_train, Y_train])

    ALL_X_test = np.concatenate([ALL_X_test, X_test])
    ALL_Y_test = np.concatenate([ALL_Y_test, Y_test])

    return ALL_X_train, ALL_Y_train, ALL_X_test, ALL_Y_test


def train():
    generator = ModelGenerator()

    ALL_X_train, ALL_Y_train, ALL_X_test, ALL_Y_test = get_data_from_companies(100)

    counter = 1
    collected_metrics = {'RMSE': [], 'MAPE': []}
    for model in generator.generate_models():
        model.train_model(ALL_X_train, ALL_Y_train, 10)
        y_pred, metrics = model.test_model(ALL_X_test, ALL_Y_test)
        # y1 = dataset.scaler.inverse_transform(y_pred[:365])[:,4]
        # print(y1.shape)
        # y2 = dataset.scaler.inverse_transform(ALL_Y_test[:365])[:,4]
        # draw_plot(y1,y2)
        for key in collected_metrics.keys():
            collected_metrics[key].append(metrics[key])
        if counter % 12 == 0:
            print("Saving")
            JSON_obj = calculate_all_metrics(**collected_metrics)
            JSON_obj['architecture'] = model.architecture
            JSON_obj['learning_rate'] = model.learning_rate
            JSON_obj['optimizer'] = model.optimizer
            save_data_model_json('my7.json', JSON_obj)
            collected_metrics = {'RMSE': [], 'MAPE': []}
        counter += 1


def calculate_all_metrics(RMSE, MAPE):
    RMSE = np.array(RMSE)
    MAPE = np.array(MAPE)

    max_rmse = np.max(RMSE)
    min_rmse = np.min(RMSE)
    avg_rmse = np.average(RMSE)
    mean_rmse = np.mean(RMSE)
    std_rmse = np.std(RMSE)

    max_mape = np.max(MAPE)
    min_mape = np.min(MAPE)
    avg_mape = np.average(MAPE)
    mean_mape = np.mean(MAPE)
    std_mape = np.std(MAPE)

    return {
        'MAPE': {
            'MAX': str(max_mape),
            'MIN': str(min_mape),
            'AVG': str(avg_mape),
            'STD': str(std_mape),
            'MEAN': str(mean_mape),
            'VALUES': [str(x) for x in MAPE],
        },
        'RMSE': {
            'MAX': str(max_rmse),
            'MIN': str(min_rmse),
            'AVG': str(avg_rmse),
            'STD': str(std_rmse),
            'MEAN': str(mean_rmse),
            'VALUES': [str(x) for x in RMSE],
        }
    }


def create_file(path: str):
    with open(path, mode='a') as file:
        json.dump([], file, indent=4)


def save_data_model_json(path: str, data):
    with open(path, mode='r+') as file:
        content = json.load(file)
        content.append(data)
        file.seek(0)
        json.dump(content, file, indent=4)


def draw_result():
    # pred_model = load_model('my_model500.h5')
    pred_model = load_model('my_model50x50.h5')
    companyRep = CompanyRep(db)

    companies = companyRep.get_all_company_symbol()
    ALL_data = np.empty(shape=(0, 6))

    for company in companies[:150]:
        data = load_data(company.id)
        ALL_data = np.concatenate([ALL_data, data])

    dataset = DataSet(ALL_data)

    company = companies[1]
    data = load_data(company.id)

    X_train, Y_train, X_test, Y_test = dataset.preprocess(data, 0.2)

    Y_pred = pred_model.predict(X_test)

    Y_pred = dataset.scaler.inverse_transform(Y_pred)
    data = data[500:]

    Dates = [(row[0], float(row[5])) for row in data]
    Dates.sort()

    title = company.name
    draw_plot(title, Dates, Y_pred[:, 4])


def main():
    model = StockPrediction()
    model.model.summary()
    ALL_X_train, ALL_Y_train, ALL_X_test, ALL_Y_test = get_data_from_companies(150)
    model.train_model(ALL_X_train, ALL_Y_train, 4)
    y_pred = model.test_model(ALL_X_test, ALL_Y_test)
    # print(dataset.scaler.inverse_transform(y_pred))
    # print(dataset.scaler.inverse_transform(y_test))
    #model.save_model('my_model150xw100.h5')


def test_model():
    pred_model = load_model('my_model150xw100.h5')
    companyRep = CompanyRep(db)

    companies = companyRep.get_all_company_symbol()
    ALL_data = np.empty(shape=(0, 6))

    for company in companies[:150]:
        data = load_data(company.id)
        ALL_data = np.concatenate([ALL_data, data])

    dataset = DataSet(ALL_data)

    company = companies[1]
    data = load_data(company.id)

    test_data = data[len(data) - 2*window_size:len(data) - window_size]

    test_data = [[x[1], x[2], x[3], x[4], x[5]] for x in test_data]

    true_data = [x[5] for x in data[-window_size:]]
    test_data = dataset.scaler.transform(test_data)
    test_data = test_data[None,:]
    result = []

    for i in range(window_size):
        predictions = pred_model.predict(test_data)
        test_data = np.append(test_data, predictions[None, :], axis=1)
        test_data = test_data[:, 1:, :]
        result.append(*predictions)

    transformed_pred = dataset.scaler.inverse_transform(result)

    plt.plot([i for i in range(window_size)], true_data,label='True data')
    plt.plot([i for i in range(window_size)],transformed_pred[:, 4],label='predicted')
    plt.show()


if __name__ == '__main__':
    window_size = 100
    plt.rcParams.update({'font.size': 22})
    db = Database(DBConfig())
    # draw_result()
    #main()
    test_model()
