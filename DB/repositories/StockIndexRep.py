from sqlalchemy.orm import Session

from DB.Db import Database
from DB.models.StockIndex import StockIndex


class StockRep:
    __db = None

    def __init__(self, db: Database):
        self.__db = db

    def add_stock_index(self, instance: StockIndex):
        session = Session(bind=self.__db.get_engine())
        session.add(instance)
        session.commit()
        session.close()

    def get_stock_index(self, symbol_id=None, datetime_begin=None, datetime_end=None):
        session = Session(bind=self.__db.get_engine())
        query = session.query(StockIndex.datetime, StockIndex.open_val, StockIndex.high_val,
                              StockIndex.low_val, StockIndex.volume_val, StockIndex.close_val)
        if symbol_id:
            query = query.filter(StockIndex.symbol_id == symbol_id)
        if datetime_begin:
            query = query.filter(StockIndex.datetime >= datetime_begin)
        if datetime_end:
            query = query.filter(StockIndex.datetime <= datetime_end)

        instances = query.all()
        session.close()
        return instances

    def update_stock_index(self, name, new_instance: StockIndex):
        session = Session(bind=self.__db.get_engine())
        obj = session.query(StockIndex).filter(StockIndex.symbol == name).first()
        obj.datetime = new_instance.datetime
        obj.open_val = new_instance.open_val
        obj.low_val = new_instance.low_val
        obj.adjclose_val = new_instance.adjclose_val
        obj.close_val = new_instance.close_val
        obj.high_val = new_instance.high_val
        obj.volume_val = new_instance.volume_val
        session.add(obj)
        session.commit()
        session.close()

    def delete_stock_index(self, name):
        session = Session(bind=self.__db.get_engine())
        obj = session.query(StockIndex).filter(StockIndex.symbol == name).first()
        session.delete(obj)
        session.commit()
        session.close()
