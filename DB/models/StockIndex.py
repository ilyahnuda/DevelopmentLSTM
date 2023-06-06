from sqlalchemy import Integer, String, \
    Column, DateTime, DECIMAL
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class StockIndex(Base):
    __tablename__ = 'stocks_stockindex'

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, nullable=False)
    datetime = Column(DateTime, nullable=False)
    open_val = Column(DECIMAL, nullable=False)
    high_val = Column(DECIMAL, nullable=False)
    low_val = Column(DECIMAL, nullable=False)
    close_val = Column(DECIMAL, nullable=False)
    adjclose_val = Column(DECIMAL, nullable=False)
    volume_val = Column(DECIMAL, nullable=False)


