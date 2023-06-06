class DBConfig:
    def __init__(self):
        self.driver = 'psycopg2'
        self.dialect = 'postgresql'
        self.username = 'postgres'
        self.password = '123456'
        self.host = 'localhost'
        self.port = '5432'
        self.db = 'MarketDB'