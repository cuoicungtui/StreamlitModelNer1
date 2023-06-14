from pymongo import MongoClient

# def connect_mongo(path__atlat):
#     try:
#         client = MongoClient(path__atlat)
#         db = client['AIDATA']
#         collection = db['data1']
#         # print("Kết nối thành công tới MongoDB Atlas")
#         return collection
#     except Exception as e:
#         print("Lỗi khi kết nối tới MongoDB Atlas:", e)

class dataMongo:
    def __init__(self, path_atlas):
        self.path_atlas = path_atlas
        self.collection = self.connect()
    def connect(self):
        try:
            client = MongoClient(self.path_atlas)
            db = client['AIDATA']
            collection = db['data1']
            print("Kết nối thành công tới MongoDB Atlas")
            return collection
        except Exception as e:
            print("Lỗi khi kết nối tới MongoDB Atlas:", e)

    def insert(self, data):
        try:
            data = {'Sequence':data}
            self.collection.insert_one(data)
            print("Insert data thành công vào MongoDB Atlas")
        except Exception as e:
            print("Lỗi khi insert data vào MongoDB Atlas:", e)
    def find(self, data):
        return self.collection.find(data)
    