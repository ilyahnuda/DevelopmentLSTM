from sqlalchemy import create_engine

from DB.DBConfig import DBConfig


class Database:
    __engine = None

    def __init__(self, conf: DBConfig):
        self.__engine = create_engine(f'{conf.dialect}+{conf.driver}://'
                                 f'{conf.username}:{conf.password}@{conf.host}:{conf.port}/{conf.db}')

    def get_engine(self):
        return self.__engine
