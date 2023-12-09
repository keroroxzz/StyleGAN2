import time

class PerfMeter():
    def __init__(self):
        self.dict={}
        
    def add(self, key, value):
        if not key in self.dict.keys():
            self.dict[key]=[value,1,0]
        else:
            pvalue, count, time = self.dict[key]
            self.dict[key]=[pvalue+value,count+1,time]
        
    def getAvg(self, key):
        return self.dict[key][0]/self.dict[key][1]
    
    def beg(self, key):
        if not key in self.dict.keys():
            self.dict[key]=[0,0,0]
        self.dict[key][2]=time.time()
    
    def end(self, key):
        self.add(key, time.time()-self.dict[key][2])
        
    def print(self):
        for key in self.dict.keys():
            print(f'{key}:\t\t {self.getAvg(key)}')
            
    def printRank(self):
        pairs=[]
        for key in self.dict.keys():
            pairs.append((key,self.getAvg(key)))
        pairs.sort(reverse = True, key = lambda s: s[1])
        
        for (key, time) in pairs:
            print(f'{key}:\t {time}')

pm = PerfMeter()

def measure(arg):
    def decorator(func):
        def wrapper(*args, **kwargs):
            x = func(*args, **kwargs)
            return x
        return wrapper
    return decorator