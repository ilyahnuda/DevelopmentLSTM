import sqlalchemy

from sqlalchemy import Integer, String, \
    Column, DateTime, DECIMAL
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class Company(Base):
    __tablename__ = 'stocks_company'

    id = Column(Integer, primary_key=True,autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    symbol = Column(String(10), unique=True)
    weight = Column(DECIMAL, nullable=False)
    price = Column(DECIMAL, nullable=False)
    chg = Column(DECIMAL, nullable=False)
    percent_chg = Column(DECIMAL, nullable=False)